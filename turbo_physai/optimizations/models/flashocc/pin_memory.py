"""Enable pinned host memory for MMDetection DataLoader instances."""

import functools


def pin_memory_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        kwargs["pin_memory"] = True
        return original(*args, **kwargs)

    return wrapped
