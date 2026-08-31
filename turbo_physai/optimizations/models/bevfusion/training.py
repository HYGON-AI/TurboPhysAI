# Copyright 2018-2019 OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""BEVFusion training-time data movement and runtime policy wrappers."""

from collections import OrderedDict
import functools
import os
import threading


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _option_flag(options, name, env_name, default=False):
    if name in options:
        return bool(options[name])
    return _env_flag(env_name, default)


def parse_losses(self, losses):
    """Reduce all scalar losses with one collective and one host transfer."""

    del self
    import torch
    import torch.distributed as dist

    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(item.mean() for item in loss_value)
        else:
            raise TypeError(
                f"{loss_name} is not a tensor or list of tensors"
            )

    loss = sum(value for name, value in log_vars.items() if "loss" in name)
    log_vars["loss"] = loss
    names = list(log_vars)
    reduced = torch.stack([log_vars[name].detach() for name in names])
    if dist.is_available() and dist.is_initialized():
        reduced.div_(dist.get_world_size())
        dist.all_reduce(reduced)
    return loss, OrderedDict(zip(names, reduced.cpu().tolist()))


class CudaPrefetchLoader:
    """Prefetch non-cpu-only DataContainer values on a dedicated stream."""

    def __init__(self, loader, device=None, bev_meta_cfg=None):
        import torch

        self.loader = loader
        self.device = torch.device(
            "cuda",
            torch.cuda.current_device() if device is None else device,
        )
        self.stream = torch.cuda.Stream(device=self.device)
        self.bev_meta_cfg = bev_meta_cfg
        self._bev_ref_points = None
        self._loader_iter = None
        self._current = None
        self._prefetched = None
        self._worker = None
        self._worker_error = None
        self._exhausted = False

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        self._loader_iter = iter(self.loader)
        self._current = self._load_next()
        self._prefetched = None
        self._worker = None
        self._worker_error = None
        self._exhausted = False
        if self._current is not None:
            self._launch_preload()
        return self

    def __next__(self):
        import torch

        if self._current is None and not self._promote_prefetched():
            raise StopIteration
        current_stream = torch.cuda.current_stream(device=self.device)
        batch, ready_event = self._current
        current_stream.wait_event(ready_event)
        self._record_stream(batch, current_stream)
        self._current = None
        return batch

    def _launch_preload(self):
        self._worker_error = None
        self._worker = threading.Thread(target=self._background_preload)
        self._worker.daemon = True
        self._worker.start()

    def _background_preload(self):
        try:
            self._prefetched = self._load_next()
            self._exhausted = self._prefetched is None
        except Exception as exc:  # pragma: no cover - surfaced on join
            self._worker_error = exc
            self._prefetched = None
            self._exhausted = True

    def _promote_prefetched(self):
        if self._worker is not None:
            self._worker.join()
            self._worker = None
        if self._worker_error is not None:
            raise RuntimeError("CUDA prefetch worker failed") from self._worker_error
        self._current = self._prefetched
        self._prefetched = None
        if self._current is None:
            return False
        self._launch_preload()
        return True

    def _load_next(self):
        import torch

        try:
            batch = next(self._loader_iter)
        except StopIteration:
            return None
        with torch.cuda.stream(self.stream):
            with torch.cuda.device(self.device):
                batch = self._to_cuda(self._pin_memory(batch))
                if self.bev_meta_cfg is not None:
                    self._prepare_bev_geometry(batch)
                ready_event = torch.cuda.Event()
                ready_event.record(self.stream)
        return batch, ready_event

    @staticmethod
    def _data_container_type():
        from mmcv.parallel.data_container import DataContainer

        return DataContainer

    def _pin_memory(self, obj):
        import torch

        data_container = self._data_container_type()
        if isinstance(obj, data_container):
            if obj.cpu_only:
                return obj
            return data_container(
                self._pin_memory(obj.data),
                stack=obj.stack,
                padding_value=obj.padding_value,
                cpu_only=False,
                pad_dims=obj.pad_dims,
            )
        if torch.is_tensor(obj):
            if obj.device.type != "cpu" or obj.is_pinned():
                return obj
            return obj.pin_memory()
        if isinstance(obj, list):
            return [self._pin_memory(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._pin_memory(item) for item in obj)
        if isinstance(obj, dict):
            return type(obj)(
                (key, self._pin_memory(value)) for key, value in obj.items()
            )
        return obj

    def _to_cuda(self, obj):
        import torch

        data_container = self._data_container_type()
        if isinstance(obj, data_container):
            if obj.cpu_only:
                return obj
            return data_container(
                self._to_cuda(obj.data),
                stack=obj.stack,
                padding_value=obj.padding_value,
                cpu_only=False,
                pad_dims=obj.pad_dims,
            )
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, list):
            return [self._to_cuda(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_cuda(item) for item in obj)
        if isinstance(obj, dict):
            return type(obj)(
                (key, self._to_cuda(value)) for key, value in obj.items()
            )
        return obj

    def _record_stream(self, obj, stream):
        import torch

        data_container = self._data_container_type()
        if isinstance(obj, data_container):
            self._record_stream(obj.data, stream)
        elif torch.is_tensor(obj):
            obj.record_stream(stream)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._record_stream(item, stream)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._record_stream(value, stream)

    def _prepare_bev_geometry(self, batch):
        if os.getenv("ENABLE_BEV_META_PREFETCH", "1") != "1":
            return
        if not isinstance(batch, dict) or "img_metas" not in batch:
            return

        data_container = self._data_container_type()
        img_metas = batch["img_metas"]
        data = img_metas.data if isinstance(img_metas, data_container) else img_metas
        for meta_group in self._collect_meta_groups(data):
            self._prepare_bev_meta_group(meta_group)

    def _collect_meta_groups(self, obj):
        if self._is_meta(obj):
            return [[obj]]

        if isinstance(obj, (list, tuple)):
            if obj and all(self._is_meta(item) for item in obj):
                return [list(obj)]

            if obj and all(isinstance(item, (list, tuple)) for item in obj):
                min_len = min((len(item) for item in obj), default=0)
                if min_len > 0 and all(
                    all(self._is_meta(item[index]) for item in obj)
                    for index in range(min_len)
                ):
                    return [
                        [item[index] for item in obj] for index in range(min_len)
                    ]

            groups = []
            for item in obj:
                groups.extend(self._collect_meta_groups(item))
            return groups

        if isinstance(obj, dict):
            groups = []
            for value in obj.values():
                groups.extend(self._collect_meta_groups(value))
            return groups

        return []

    @staticmethod
    def _is_meta(obj):
        return (
            isinstance(obj, dict)
            and "lidar2img" in obj
            and "img_shape" in obj
        )

    def _prepare_bev_meta_group(self, img_metas):
        import numpy as np
        import torch

        if not img_metas or all("_bev_indexes" in meta for meta in img_metas):
            return

        masks = []
        lengths = []
        for meta in img_metas:
            ref_cam, bev_mask, cam_query_mask, index_lengths = (
                self._compute_bev_geometry(meta)
            )
            meta["_bev_reference_points_cam"] = self._array_to_cuda(
                ref_cam, torch.float32
            )
            meta["_bev_bev_mask"] = self._array_to_cuda(bev_mask, torch.bool)
            meta["_bev_index_lengths"] = self._array_to_cuda(
                index_lengths, torch.long
            )
            masks.append(cam_query_mask)
            lengths.append(index_lengths)

        max_index_len = max(int(length.max()) for length in lengths)
        num_query = masks[0].shape[-1]
        for bucket_len in self.bev_meta_cfg["buckets"]:
            if max_index_len <= bucket_len:
                max_index_len = min(num_query, bucket_len)
                break

        for meta, mask in zip(img_metas, masks):
            indexes = np.argsort(
                -mask.astype(np.int64), axis=-1, kind="stable"
            )[:, :max_index_len]
            meta["_bev_indexes"] = self._array_to_cuda(
                np.ascontiguousarray(indexes), torch.long
            )

    def _compute_bev_geometry(self, meta):
        import numpy as np

        ref_points = self._get_bev_reference_points()
        pc_range = self.bev_meta_cfg["pc_range"]
        ref_scaled = ref_points.copy()
        ref_scaled[..., 0] = (
            ref_scaled[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        ref_scaled[..., 1] = (
            ref_scaled[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        ref_scaled[..., 2] = (
            ref_scaled[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )
        lidar2img = np.asarray(meta["lidar2img"], dtype=np.float32)
        x = ref_scaled[..., 0][None, :, :]
        y = ref_scaled[..., 1][None, :, :]
        z = ref_scaled[..., 2][None, :, :]
        cam_x = (
            lidar2img[:, 0, 0, None, None] * x
            + lidar2img[:, 0, 1, None, None] * y
            + lidar2img[:, 0, 2, None, None] * z
            + lidar2img[:, 0, 3, None, None]
        )
        cam_y = (
            lidar2img[:, 1, 0, None, None] * x
            + lidar2img[:, 1, 1, None, None] * y
            + lidar2img[:, 1, 2, None, None] * z
            + lidar2img[:, 1, 3, None, None]
        )
        cam_z = (
            lidar2img[:, 2, 0, None, None] * x
            + lidar2img[:, 2, 1, None, None] * y
            + lidar2img[:, 2, 2, None, None] * z
            + lidar2img[:, 2, 3, None, None]
        )
        cam_x = cam_x.transpose(0, 2, 1)
        cam_y = cam_y.transpose(0, 2, 1)
        cam_z = cam_z.transpose(0, 2, 1)

        img_h, img_w = self._img_hw(meta)
        eps = np.float32(1e-5)
        denom = np.maximum(cam_z, eps)
        ref_x = cam_x / denom / np.float32(img_w)
        ref_y = cam_y / denom / np.float32(img_h)
        reference_points_cam = np.stack((ref_x, ref_y), axis=-1).astype(
            np.float32, copy=False
        )

        bev_mask = (
            (cam_z > eps)
            & (ref_y > 0.0)
            & (ref_y < 1.0)
            & (ref_x < 1.0)
            & (ref_x > 0.0)
        )
        cam_query_mask = bev_mask.sum(-1) > 0
        index_lengths = cam_query_mask.sum(-1, dtype=np.int64)
        return reference_points_cam, bev_mask, cam_query_mask, index_lengths

    def _get_bev_reference_points(self):
        import numpy as np

        if self._bev_ref_points is not None:
            return self._bev_ref_points

        cfg = self.bev_meta_cfg
        height = cfg["bev_h"]
        width = cfg["bev_w"]
        z_extent = cfg["pc_range"][5] - cfg["pc_range"][2]
        depth = cfg["num_points_in_pillar"]
        zs = np.linspace(0.5, z_extent - 0.5, depth, dtype=np.float32).reshape(
            depth, 1, 1
        ) / np.float32(z_extent)
        xs = np.linspace(0.5, width - 0.5, width, dtype=np.float32).reshape(
            1, 1, width
        ) / np.float32(width)
        ys = np.linspace(0.5, height - 0.5, height, dtype=np.float32).reshape(
            1, height, 1
        ) / np.float32(height)
        zs = np.broadcast_to(zs, (depth, height, width))
        xs = np.broadcast_to(xs, (depth, height, width))
        ys = np.broadcast_to(ys, (depth, height, width))
        self._bev_ref_points = np.stack((xs, ys, zs), axis=-1).reshape(
            depth, height * width, 3
        ).astype(np.float32, copy=False)
        return self._bev_ref_points

    @staticmethod
    def _img_hw(meta):
        img_shape = meta["img_shape"]
        if isinstance(img_shape, (list, tuple)) and img_shape:
            first = img_shape[0]
            if isinstance(first, (list, tuple)):
                return int(first[0]), int(first[1])
            return int(img_shape[0]), int(img_shape[1])
        return int(img_shape[0]), int(img_shape[1])

    def _array_to_cuda(self, array, dtype):
        import numpy as np
        import torch

        tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=True)

    @staticmethod
    def build_bev_meta_prefetch_cfg(cfg):
        """Extract BEV metadata dimensions from an MMDetection config."""

        if os.getenv("ENABLE_BEV_META_PREFETCH", "1") != "1":
            return None
        try:
            head = cfg.model.pts_bbox_head
            encoder = head.transformer.encoder
            return {
                "bev_h": int(head.bev_h),
                "bev_w": int(head.bev_w),
                "pc_range": [float(value) for value in encoder.pc_range],
                "num_points_in_pillar": int(
                    encoder.get("num_points_in_pillar", 4)
                ),
                "buckets": (9728, 10240, 11264, 12288),
            }
        except Exception:
            return None


def _convert_conv2d_weights_to_channels_last(model):
    import torch

    converted = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.data.contiguous(
                memory_format=torch.channels_last
            )
            converted += 1
    return converted


def _compile_model(model, mode):
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return model
    model.encoders["camera"]["backbone"] = torch.compile(
        model.encoders["camera"]["backbone"],
        mode=mode,
        fullgraph=False,
        dynamic=False,
    )
    return model


def training_wrapper(original, options):
    """Construct the runtime recipe without modifying global state."""

    options = dict(options)

    @functools.wraps(original)
    def wrapped(model, dataset, cfg, *args, **kwargs):
        import torch

        start_method = str(
            options.get(
                "start_method",
                os.getenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "fork"),
            )
        )
        if start_method:
            current = torch.multiprocessing.get_start_method(allow_none=True)
            if current != start_method:
                torch.multiprocessing.set_start_method(
                    start_method, force=current is not None
                )

        if _option_flag(
            options, "channels_last", "MMDET3D_CHANNELS_LAST", True
        ):
            converted = _convert_conv2d_weights_to_channels_last(model)
            logger = getattr(cfg, "logger", None)
            if logger is not None:
                logger.info("converted Conv2d weights to channels-last: %d", converted)

        compile_mode = str(
            options.get("compile_mode", "max-autotune-no-cudagraphs")
        )
        model = _compile_model(model, compile_mode)

        enable_prefetch = (
            torch.cuda.is_available()
            and _option_flag(
                options, "cuda_prefetch", "ENABLE_CUDA_PREFETCH", True
            )
        )
        if not enable_prefetch:
            return original(model, dataset, cfg, *args, **kwargs)

        original_globals = original.__globals__
        if "build_dataloader" not in original_globals:
            raise RuntimeError(
                "TurboPhysAI BEVFusion training wrapper cannot locate "
                "train_model.build_dataloader"
            )
        build_dataloader = original_globals["build_dataloader"]
        bev_meta_cfg = CudaPrefetchLoader.build_bev_meta_prefetch_cfg(cfg)
        # train_model builds all training loaders first and the validation
        # loader later. EvalHook requires the latter to remain an actual
        # torch DataLoader, matching the optimized reference implementation
        # which wraps only ``data_loaders``.
        remaining_training_loaders = [
            len(dataset) if isinstance(dataset, (list, tuple)) else 1
        ]

        @functools.wraps(build_dataloader)
        def build_prefetch_loader(*builder_args, **builder_kwargs):
            loader = build_dataloader(*builder_args, **builder_kwargs)
            if remaining_training_loaders[0] <= 0:
                return loader
            remaining_training_loaders[0] -= 1
            return CudaPrefetchLoader(
                loader,
                device=torch.cuda.current_device(),
                bev_meta_cfg=bev_meta_cfg,
            )

        original_globals["build_dataloader"] = build_prefetch_loader
        try:
            return original(model, dataset, cfg, *args, **kwargs)
        finally:
            original_globals["build_dataloader"] = build_dataloader

    return wrapped
