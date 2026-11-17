# Adapted from Isaac-GR00T, SPDX-License-Identifier: Apache-2.0
import inspect
from functools import wraps
from collections.abc import MutableMapping
import torch

ENV_VAR = "groot.data.pixel_values_bf16"

class Bf16PixelValuesCollator:
    """Wrap a collator and cast ``output["inputs"]["pixel_values"]`` to BF16.

    A module-level class so it can be pickled for ``spawn``/``forkserver`` DataLoaders. Any other
    attribute (for example ``.processor``) is forwarded to the wrapped collator.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        # Only called when normal lookup fails. Guard ``inner`` itself so that copy/pickle on a
        # half-built instance does not recurse.
        if name == "inner" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __call__(self, features):
        output = self.inner(features)
        inputs = output.get("inputs") if isinstance(output, MutableMapping) else None
        if not isinstance(inputs, MutableMapping):
            raise ValueError(f"{ENV_VAR}=1: collator output has no 'inputs' mapping")
        pixel_values = inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f"{ENV_VAR}=1: inputs['pixel_values'] is missing or not a tensor")
        if pixel_values.device.type != "cpu":
            raise ValueError(
                f"{ENV_VAR}=1: inputs['pixel_values'] must be a CPU tensor, "
                f"got {pixel_values.device}"
            )
        if pixel_values.dtype == torch.float32:
            inputs["pixel_values"] = pixel_values.to(torch.bfloat16)
        elif pixel_values.dtype != torch.bfloat16:
            raise ValueError(
                f"{ENV_VAR}=1: inputs['pixel_values'] must be float32 or bfloat16, "
                f"got {pixel_values.dtype}"
            )
        return output



def trainer_init_bf16_wrapper(original, group_options):
    signature = inspect.signature(original)
    if not {"args", "data_collator"}.issubset(signature.parameters) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ):
        from transformers import Trainer
        signature = inspect.signature(Trainer.__init__)

    @wraps(original)
    def wrapped(*args, **kwargs):
        extras = {key: value for key, value in kwargs.items() if key not in signature.parameters}
        forwarded = {key: value for key, value in kwargs.items() if key in signature.parameters}
        bound = signature.bind_partial(*args, **forwarded)
        training_args = bound.arguments.get("args")
        collator = bound.arguments.get("data_collator")
        if (training_args is not None and getattr(training_args, "deepspeed", None)
            and getattr(training_args, "bf16", False) and not getattr(training_args, "fp16", False)
            and collator is not None and not isinstance(collator, Bf16PixelValuesCollator)):
            bound.arguments["data_collator"] = Bf16PixelValuesCollator(collator)
        return original(*bound.args, **bound.kwargs, **extras)
    return wrapped
