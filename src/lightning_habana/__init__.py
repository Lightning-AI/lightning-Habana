"""Root package info."""

import os

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from lightning_utilities.core.imports import package_available  # noqa: E402

_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")
if _HABANA_FRAMEWORK_AVAILABLE:
    from habana_frameworks.torch.utils.library_loader import is_habana_available

    _HPU_AVAILABLE: bool = is_habana_available()
else:
    _HPU_AVAILABLE = False

from lightning_habana.__about__ import *  # noqa: E402, F401, F403
from lightning_habana.accelerator import HPUAccelerator  # noqa: E402
from lightning_habana.plugins.io_plugin import HPUCheckpointIO  # noqa: E402
from lightning_habana.plugins.precision import HPUPrecisionPlugin  # noqa: E402
from lightning_habana.strategies.parallel import HPUParallelStrategy  # noqa: E402
from lightning_habana.strategies.single import SingleHPUStrategy  # noqa: E402

__all__ = [
    "HPUAccelerator",
    "HPUParallelStrategy",
    "SingleHPUStrategy",
    "HPUPrecisionPlugin",
    "HPUCheckpointIO",
]
