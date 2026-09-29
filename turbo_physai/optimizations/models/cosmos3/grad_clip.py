# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from collections import defaultdict
import torch

try:  # torch >= 2.4 public path (target: torch 2.10 HIP)
    from torch.distributed.tensor import DTensor
except ImportError:  # older local dev environments
    from torch.distributed._tensor import DTensor
try:
    # Share canonical streams with an existing HCU FSDP adaptation.
    from cosmos_framework.utils.fsdp_ordered_comm import materialize_dtensor_norm
except ModuleNotFoundError as exc:
    if exc.name != "cosmos_framework.utils.fsdp_ordered_comm":
        raise
    from ...common.pytorch.fsdp_ordered_comm import materialize_dtensor_norm


@torch.no_grad()
def _clip_grad(
    parameters: list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    return_norm_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Params are grouped by their ``device_mesh`` (by mesh-dim-names string —
    plain (non-DTensor) params map to ``"default"``). A scalar L2 norm is
    computed per mesh group, DTensor results are reduced to local scalars
    via ``.full_tensor()``, the per-mesh scalars are combined into one
    global norm, and (unless ``return_norm_only=True``) every mesh group
    is rescaled with that single global scalar.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        return_norm_only: if True, skip in-place rescaling of grads and only
            return the computed norms.

    Returns:
        ``(total_norm, per_mesh_norms)`` where ``total_norm`` is the global
        scalar norm used for the rescale, and ``per_mesh_norms`` maps each
        mesh-dim-names key (or ``"default"`` for plain params) to its
        pre-clip per-mesh L2 norm.

    """
    # Group the parameters by their device meshes.
    parameters_by_mesh: dict[str, list[torch.Tensor]] = defaultdict(list)
    for param in parameters:
        if param.grad is None:
            raise ValueError(
                f"_clip_grad received a parameter with no gradient "
                f"(shape={tuple(param.shape)}, dtype={param.dtype}); "
                "callers are expected to pre-filter."
            )

        # If one parameter belongs to multiple meshes, use a flattened mesh name
        # by concatenating all the mesh-dim names together.  ``mesh_dim_names``
        # is ``tuple[str, ...] | None`` on DeviceMesh — fall back to ``default``
        # when names weren't assigned.
        if hasattr(param, "device_mesh"):
            names = param.device_mesh.mesh_dim_names
            device_mesh_str = "-".join(names) if names else "default"
        else:
            device_mesh_str = "default"
        parameters_by_mesh[device_mesh_str].append(param)

    # Compute the norm for each mesh group
    per_mesh_norms: dict[str, torch.Tensor] = {}
    per_mesh_norm_list = []
    for mesh, params in parameters_by_mesh.items():
        # Every param reached here passed the ``param.grad is None`` check in
        # the grouping loop above, so this list comprehension is total.
        grads = [p.grad for p in params]
        mesh_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

        # If mesh_norm is a DTensor, the placements must be
        # `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this
        # tensor's process group and then convert it to a local tensor.
        # NOTE: It has two purposes:
        # 1. to make sure the total norm is computed correctly when PP is used (see below)
        # 2. to return a reduced mesh_norm tensor whose .item() would return the correct value
        if isinstance(mesh_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, mesh_norm will be a local tensor.

            # Remove FT replicate dimension if it exists.
            if torch.version.hip is not None:
                mesh_norm = materialize_dtensor_norm(mesh_norm)
            else:
                mesh_norm = mesh_norm.full_tensor()
        # Expose the (rank-replicated) per-mesh scalar for diagnostic logging.
        per_mesh_norms[mesh] = mesh_norm

        # Make the norm to be a 1D tensor so we can call cat() later.
        if mesh_norm.ndim == 0:
            mesh_norm = mesh_norm.reshape(1)
        per_mesh_norm_list.append(mesh_norm)

    # Compute the total norm among all meshes.
    if len(per_mesh_norm_list) > 1:
        per_mesh_norm_tensor = torch.cat(per_mesh_norm_list)
        if math.isinf(norm_type):
            total_norm = torch.max(per_mesh_norm_tensor)
        else:
            per_mesh_norm_tensor **= norm_type
            total_norm = torch.sum(per_mesh_norm_tensor)
            total_norm **= 1.0 / norm_type
    else:
        assert per_mesh_norm_list[0].numel() == 1, "total_norm should be a scalar"
        total_norm = per_mesh_norm_list[0].view(-1)[0]

    if not return_norm_only:
        # Perform clipping on each mesh group
        for mesh, params in parameters_by_mesh.items():
            torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach)

    return total_norm, per_mesh_norms
