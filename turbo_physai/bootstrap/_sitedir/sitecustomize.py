# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Interpreter startup hook for TurboPhysAI.

``site`` imports the first ``sitecustomize`` found on ``sys.path``. Because
``turbo-physai run`` prepends this directory to ``PYTHONPATH``, that is this
file -- which would otherwise shadow a ``sitecustomize`` the user already has.
So chain to theirs first, then activate.
"""

import os
import sys


def _run_user_sitecustomize():
    """Import the sitecustomize this file shadows, if the user has one."""

    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    remaining = [
        entry
        for entry in sys.path
        if entry and os.path.abspath(entry) != here
    ]
    spec = importlib.machinery.PathFinder().find_spec("sitecustomize", remaining)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["sitecustomize"] = module
    spec.loader.exec_module(module)


try:
    _run_user_sitecustomize()
except Exception:  # noqa: BLE001 - a broken user hook must not mask ours
    import traceback

    traceback.print_exc()

if os.environ.get("TURBO_PHYSAI_BOOTSTRAP") == "1":
    try:
        from turbo_physai.bootstrap import activate
    except BaseException as error:
        # Importing turbo_physai can fail on its own (e.g. it is not installed
        # in this interpreter), which would leave activate() -- and therefore
        # its own abort guard -- unreachable. site.execsitecustomize() only
        # prints the exception, so training would silently run unoptimized.
        import traceback

        traceback.print_exc()
        sys.stderr.write(
            "turbo-physai: cannot import turbo_physai in this interpreter "
            "(%s); aborting instead of training unoptimized\n" % (error,)
        )
        sys.stderr.flush()
        os._exit(91)
    activate()
