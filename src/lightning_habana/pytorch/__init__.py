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

from lightning_habana.__about__ import *  # noqa: E402, F401, F403
from lightning_habana.pytorch.accelerator import HPUAccelerator  # noqa: E402
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO  # noqa: E402
from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin  # noqa: E402
from lightning_habana.pytorch.strategies.parallel import HPUParallelStrategy  # noqa: E402
from lightning_habana.pytorch.strategies.single import SingleHPUStrategy  # noqa: E402

__all__ = [
    "HPUAccelerator",
    "HPUParallelStrategy",
    "SingleHPUStrategy",
    "HPUPrecisionPlugin",
    "HPUCheckpointIO",
]
