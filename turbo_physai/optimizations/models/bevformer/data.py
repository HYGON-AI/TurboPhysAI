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

"""BEVFormer DataLoader and asynchronous geometry prefetch replacement."""

import os
import random
import threading
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import get_dist_info
from mmdet.datasets.samplers import GroupSampler
from torch.utils.data import DataLoader

class CudaPrefetchLoader:
    """Prefetch non-cpu_only batch data onto the current CUDA device."""

    def __init__(self, loader, device=None, bev_meta_cfg=None):
        self.loader = loader
        self.device = torch.device(
            'cuda',
            torch.cuda.current_device() if device is None else device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.bev_meta_cfg = bev_meta_cfg
        self._bev_ref_points = None
        self._loader_iter = None
        self._current = None
        self._prefetched = None
        self._worker = None
        self._worker_error = None
        self._exhausted = False

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

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __next__(self):
        if self._current is None:
            if not self._promote_prefetched():
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
            raise RuntimeError('CUDA prefetch worker failed') from self._worker_error

        self._current = self._prefetched
        self._prefetched = None
        if self._current is None:
            return False

        self._launch_preload()
        return True

    def _load_next(self):
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

    def _pin_memory(self, obj):
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj
            return DataContainer(
                self._pin_memory(obj.data),
                stack=obj.stack,
                padding_value=obj.padding_value,
                cpu_only=False,
                pad_dims=obj.pad_dims)

        if torch.is_tensor(obj):
            return obj if obj.is_pinned() else obj.pin_memory()

        if isinstance(obj, list):
            return [self._pin_memory(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._pin_memory(item) for item in obj)

        if isinstance(obj, dict):
            return type(obj)((key, self._pin_memory(value))
                             for key, value in obj.items())

        return obj

    def _to_cuda(self, obj):
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj
            return DataContainer(
                self._to_cuda(obj.data),
                stack=obj.stack,
                padding_value=obj.padding_value,
                cpu_only=False,
                pad_dims=obj.pad_dims)

        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)

        if isinstance(obj, list):
            return [self._to_cuda(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._to_cuda(item) for item in obj)

        if isinstance(obj, dict):
            return type(obj)((key, self._to_cuda(value))
                             for key, value in obj.items())

        return obj

    def _record_stream(self, obj, stream):
        if isinstance(obj, DataContainer):
            self._record_stream(obj.data, stream)
            return

        if torch.is_tensor(obj):
            obj.record_stream(stream)
            return

        if isinstance(obj, list):
            for item in obj:
                self._record_stream(item, stream)
            return

        if isinstance(obj, tuple):
            for item in obj:
                self._record_stream(item, stream)
            return

        if isinstance(obj, dict):
            for value in obj.values():
                self._record_stream(value, stream)

    def _prepare_bev_geometry(self, batch):
        if os.getenv('ENABLE_BEV_META_PREFETCH', '1') != '1':
            return
        if not isinstance(batch, dict) or 'img_metas' not in batch:
            return

        data = batch['img_metas'].data if isinstance(
            batch['img_metas'], DataContainer) else batch['img_metas']
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
                        all(self._is_meta(item[i]) for item in obj)
                        for i in range(min_len)):
                    return [[item[i] for item in obj] for i in range(min_len)]

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
        return isinstance(obj, dict) and 'lidar2img' in obj and 'img_shape' in obj

    def _prepare_bev_meta_group(self, img_metas):
        if not img_metas or all('_bev_indexes' in meta for meta in img_metas):
            return

        masks = []
        lengths = []
        for meta in img_metas:
            ref_cam, bev_mask, cam_query_mask, index_lengths = \
                self._compute_bev_geometry(meta)
            meta['_bev_reference_points_cam'] = self._array_to_cuda(
                ref_cam, torch.float32)
            meta['_bev_bev_mask'] = self._array_to_cuda(bev_mask, torch.bool)
            meta['_bev_index_lengths'] = self._array_to_cuda(
                index_lengths, torch.long)
            masks.append(cam_query_mask)
            lengths.append(index_lengths)

        max_index_len = max(int(length.max()) for length in lengths)
        num_query = masks[0].shape[-1]
        for bucket_len in self.bev_meta_cfg['buckets']:
            if max_index_len <= bucket_len:
                max_index_len = min(num_query, bucket_len)
                break

        for meta, mask in zip(img_metas, masks):
            indexes = np.argsort(
                -mask.astype(np.int64), axis=-1, kind='stable')[:, :max_index_len]
            meta['_bev_indexes'] = self._array_to_cuda(
                np.ascontiguousarray(indexes), torch.long)

    def _compute_bev_geometry(self, meta):
        ref_points = self._get_bev_reference_points()
        pc_range = self.bev_meta_cfg['pc_range']
        ref_scaled = ref_points.copy()
        ref_scaled[..., 0] = ref_scaled[..., 0] * (
            pc_range[3] - pc_range[0]) + pc_range[0]
        ref_scaled[..., 1] = ref_scaled[..., 1] * (
            pc_range[4] - pc_range[1]) + pc_range[1]
        ref_scaled[..., 2] = ref_scaled[..., 2] * (
            pc_range[5] - pc_range[2]) + pc_range[2]
        lidar2img = np.asarray(meta['lidar2img'], dtype=np.float32)
        x = ref_scaled[..., 0][None, :, :]
        y = ref_scaled[..., 1][None, :, :]
        z = ref_scaled[..., 2][None, :, :]
        cam_x = (lidar2img[:, 0, 0, None, None] * x
                 + lidar2img[:, 0, 1, None, None] * y
                 + lidar2img[:, 0, 2, None, None] * z
                 + lidar2img[:, 0, 3, None, None])
        cam_y = (lidar2img[:, 1, 0, None, None] * x
                 + lidar2img[:, 1, 1, None, None] * y
                 + lidar2img[:, 1, 2, None, None] * z
                 + lidar2img[:, 1, 3, None, None])
        cam_z = (lidar2img[:, 2, 0, None, None] * x
                 + lidar2img[:, 2, 1, None, None] * y
                 + lidar2img[:, 2, 2, None, None] * z
                 + lidar2img[:, 2, 3, None, None])
        cam_x = cam_x.transpose(0, 2, 1)
        cam_y = cam_y.transpose(0, 2, 1)
        cam_z = cam_z.transpose(0, 2, 1)

        img_h, img_w = self._img_hw(meta)
        eps = np.float32(1e-5)
        denom = np.maximum(cam_z, eps)
        ref_x = cam_x / denom / np.float32(img_w)
        ref_y = cam_y / denom / np.float32(img_h)
        reference_points_cam = np.stack((ref_x, ref_y), axis=-1).astype(
            np.float32, copy=False)

        bev_mask = ((cam_z > eps) & (ref_y > 0.0) & (ref_y < 1.0)
                    & (ref_x < 1.0) & (ref_x > 0.0))
        cam_query_mask = bev_mask.sum(-1) > 0
        index_lengths = cam_query_mask.sum(-1, dtype=np.int64)
        return reference_points_cam, bev_mask, cam_query_mask, index_lengths

    def _get_bev_reference_points(self):
        if self._bev_ref_points is not None:
            return self._bev_ref_points

        cfg = self.bev_meta_cfg
        h = cfg['bev_h']
        w = cfg['bev_w']
        z = cfg['pc_range'][5] - cfg['pc_range'][2]
        d = cfg['num_points_in_pillar']
        zs = np.linspace(0.5, z - 0.5, d, dtype=np.float32).reshape(
            d, 1, 1) / np.float32(z)
        xs = np.linspace(0.5, w - 0.5, w, dtype=np.float32).reshape(
            1, 1, w) / np.float32(w)
        ys = np.linspace(0.5, h - 0.5, h, dtype=np.float32).reshape(
            1, h, 1) / np.float32(h)
        zs = np.broadcast_to(zs, (d, h, w))
        xs = np.broadcast_to(xs, (d, h, w))
        ys = np.broadcast_to(ys, (d, h, w))
        self._bev_ref_points = np.stack((xs, ys, zs), axis=-1).reshape(
            d, h * w, 3).astype(np.float32, copy=False)
        return self._bev_ref_points

    @staticmethod
    def _img_hw(meta):
        img_shape = meta['img_shape']
        if isinstance(img_shape, (list, tuple)) and img_shape:
            first = img_shape[0]
            if isinstance(first, (list, tuple)):
                return int(first[0]), int(first[1])
            return int(img_shape[0]), int(img_shape[1])
        return int(img_shape[0]), int(img_shape[1])

    def _array_to_cuda(self, array, dtype):
        tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=True)


def _build_bev_meta_prefetch_cfg(cfg):
    if os.getenv('ENABLE_BEV_META_PREFETCH', '1') != '1':
        return None
    try:
        head = cfg.model.pts_bbox_head
        encoder = head.transformer.encoder
        return dict(
            bev_h=int(head.bev_h),
            bev_w=int(head.bev_w),
            pc_range=[float(v) for v in encoder.pc_range],
            num_points_in_pillar=int(
                encoder.get('num_points_in_pillar', 4)),
            buckets=(9728, 10240, 11264, 12288))
    except Exception:
        return None



def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1,
                     dist=True, shuffle=True, seed=None,
                     shuffler_sampler=None, nonshuffler_sampler=None, **kwargs):
    from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

    # nuScenes DetectionConfig stores this as a dict_keys view, which cannot
    # be pickled when DataLoader uses spawned workers. Preserve its order and
    # values while making the dataset worker-safe.
    eval_cfg = getattr(dataset, "eval_detection_configs", None)
    class_names = getattr(eval_cfg, "class_names", None)
    if type(class_names).__name__ == "dict_keys":
        eval_cfg.class_names = list(class_names)

    rank, world_size = get_dist_info()
    if dist:
        sampler_cfg = shuffler_sampler if shuffle else nonshuffler_sampler
        if sampler_cfg is None:
            sampler_cfg = dict(type=("DistributedGroupSampler" if shuffle
                                     else "DistributedSampler"))
        sampler_args = dict(dataset=dataset, num_replicas=world_size,
                            rank=rank, seed=seed)
        if shuffle:
            sampler_args["samples_per_gpu"] = samples_per_gpu
        else:
            sampler_args["shuffle"] = False
        sampler = build_sampler(sampler_cfg, sampler_args)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = (partial(worker_init_fn, num_workers=num_workers, rank=rank,
                       seed=seed) if seed is not None else None)
    loader_kwargs = dict(kwargs)
    loader_kwargs.setdefault("pin_memory", True)
    if num_workers > 0:
        loader_kwargs.setdefault("prefetch_factor", 16)
        loader_kwargs.setdefault("persistent_workers", True)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn, **loader_kwargs)
    # MMCV's EvalHook requires the validation loader to be an actual
    # torch.utils.data.DataLoader. Only the shuffled training loader should be
    # wrapped with CUDA prefetching; validation/test loaders use shuffle=False.
    if (shuffle and torch.cuda.is_available()
            and os.getenv("TURBO_PHYSAI_CUDA_PREFETCH", "1") == "1"):
        bev_meta_cfg = dict(
            bev_h=200, bev_w=200,
            pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
            num_points_in_pillar=4,
            buckets=(9728, 10240, 11264, 12288),
        )
        return CudaPrefetchLoader(loader, bev_meta_cfg=bev_meta_cfg)
    return loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
