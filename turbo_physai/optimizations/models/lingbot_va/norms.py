"""Tensor-only Norm helpers; compile only selected model paths."""
import torch
import torch.nn.functional as F

def rms_norm(value, normalized_shape, weight, eps):
    return F.rms_norm(value, normalized_shape, weight, eps=eps)

def layer_norm_cast(value, normalized_shape, weight, bias, eps, force_float=False):
    if force_float:
        weight=weight.float() if weight is not None else None
        bias=bias.float() if bias is not None else None
    return F.layer_norm(value.float(), normalized_shape, weight, bias, eps).to(value.dtype)

def rms_adapter(compiled):
    def apply(norm, value):
        if type(norm) is not torch.nn.RMSNorm:
            return norm(value)
        return compiled(value, norm.normalized_shape, norm.weight, norm.eps)
    return apply

def layer_adapter(compiled):
    def apply(norm, value):
        from diffusers.models.normalization import FP32LayerNorm
        if type(norm) not in (torch.nn.LayerNorm, FP32LayerNorm):
            return norm(value.float()).to(value.dtype)
        return compiled(value, norm.normalized_shape, norm.weight, norm.bias, norm.eps, type(norm) is FP32LayerNorm)
    return apply
