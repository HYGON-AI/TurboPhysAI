"""Enable MMEngine's existing module compiler for PointPillars."""
from functools import wraps
import os


def compile_pointpillars_modules_wrapper(original, options):
    config = dict((options or {}).get("compile_config", {"backend": "inductor"}))

    @wraps(original)
    def wrapped(runner, target, *args, **kwargs):
        if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
            return original(runner, target, *args, **kwargs)
        missing = object()
        previous = runner.cfg.get("compile", missing)
        runner.cfg["compile"] = dict(config)
        try:
            return original(runner, target, *args, **kwargs)
        finally:
            if previous is missing:
                getattr(runner.cfg, "_cfg_dict", runner.cfg).pop("compile")
            else:
                runner.cfg["compile"] = previous
    return wrapped
