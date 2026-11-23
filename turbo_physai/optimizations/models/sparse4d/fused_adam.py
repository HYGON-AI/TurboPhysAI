"""Sparse4D FusedAdam registration.
"""

from mmcv.runner import OPTIMIZERS
from apex.optimizers import FusedAdam as _ApexFusedAdam

# Some Sparse4D revisions register FusedAdam while importing the plugin.
# Let that registration finish before replacing the registry entry.
import projects.mmdet3d_plugin  # noqa: F401


@OPTIMIZERS.register_module()
class FusedAdam(_ApexFusedAdam):
    pass
