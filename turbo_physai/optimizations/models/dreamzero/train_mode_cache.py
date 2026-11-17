"""Avoid redundant policy train/eval traversal."""

import functools


def invalidate_train_mode_cache(head):
    """Call after directly changing modes or replacing modules below a child."""
    head._cached_train_mode = None


def train_wrapper(original, options):
    del options

    @functools.wraps(original)
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        key = (mode, self.tune_diffusion_model, tuple(self.children()))
        if self.training == mode and getattr(self, "_cached_train_mode", None) == key:
            return self
        invalidate_train_mode_cache(self)
        original(self, mode)
        self.set_frozen_modules_to_eval_mode()
        self._cached_train_mode = key
        return self

    return train


def invalidate_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        invalidate_train_mode_cache(self)
        return original(self, *args, **kwargs)

    return wrapped


def set_frozen_modules_to_eval_mode(self):
    if self.training:
        frozen = [self.text_encoder, self.image_encoder, self.vae]
        if not self.tune_diffusion_model:
            frozen.append(self.model)
        for module in frozen:
            if module.training:
                module.eval()