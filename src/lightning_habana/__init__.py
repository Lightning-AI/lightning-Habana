# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import operator
import os

from lightning_utilities import compare_version

from lightning_habana.__about__ import *  # noqa: F401, F403
from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.datamodule.datamodule import HPUDataModule
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
from lightning_habana.pytorch.profiler.profiler import HPUProfiler
from lightning_habana.pytorch.strategies.deepspeed import HPUDeepSpeedStrategy
from lightning_habana.pytorch.strategies.parallel import HPUParallelStrategy
from lightning_habana.pytorch.strategies.single import SingleHPUStrategy
from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE

if compare_version("lightning", operator.lt, "2.0.0") and compare_version("pytorch_lightning", operator.lt, "2.0.0"):
    raise ImportError(
        "You are missing `lightning` or `pytorch-lightning` package or neither of them is in version 2.0+"
    )

if _HABANA_FRAMEWORK_AVAILABLE:
    from habana_frameworks.torch.utils.library_loader import is_habana_available

    HPU_AVAILABLE: bool = is_habana_available()
else:
    HPU_AVAILABLE = False

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)


__all__ = [
    "HPUAccelerator",
    "HPUDeepSpeedStrategy",
    "HPUParallelStrategy",
    "SingleHPUStrategy",
    "HPUPrecisionPlugin",
    "HPUCheckpointIO",
    "HPUDataModule",
    "HPUProfiler",
    "HPU_AVAILABLE",
]
