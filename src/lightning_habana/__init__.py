"""Root package info."""

import os

from lightning_utilities.core.imports import package_available

from lightning_habana.__about__ import *  # noqa: F401, F403
from lightning_habana.accelerator import HPUAccelerator
from lightning_habana.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.plugins.precision import HPUPrecisionPlugin
from lightning_habana.strategies.parallel import HPUParallelStrategy
from lightning_habana.strategies.single import SingleHPUStrategy

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")
if _HABANA_FRAMEWORK_AVAILABLE:
    from habana_frameworks.torch.utils.library_loader import is_habana_available

    _HPU_AVAILABLE = is_habana_available()
else:
    _HPU_AVAILABLE = False

__all__ = ["HPUAccelerator", "HPUParallelStrategy", "SingleHPUStrategy", "HPUPrecisionPlugin", "HPUCheckpointIO"]
