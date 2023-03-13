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
from lightning_habana.accelerator import AcceleratorHPU  # noqa: E402
from lightning_habana.plugins.io_plugin import HPUCheckpointIO  # noqa: E402
from lightning_habana.plugins.precision import PrecisionHPU  # noqa: E402
from lightning_habana.strategies.parallel import StrategyParallelHPU  # noqa: E402
from lightning_habana.strategies.single import StrategySingleHPU  # noqa: E402

__all__ = [
    "AcceleratorHPU",
    "StrategyParallelHPU",
    "StrategySingleHPU",
    "PrecisionHPU",
    "HPUCheckpointIO",
]
