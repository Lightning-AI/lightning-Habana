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


from lightning_habana.fabric.accelerator.hpu import HPUAccelerator
from lightning_habana.fabric.plugins.io.hpu_io import HPUCheckpointIO
from lightning_habana.fabric.plugins.precision.hpu_precision import HPUPrecision
from lightning_habana.fabric.strategies.hpu_parallel import HPUParallelStrategy
from lightning_habana.fabric.strategies.hpu_single import SingleHPUStrategy

__all__ = [
    "HPUAccelerator",
    "HPUParallelStrategy",
    "SingleHPUStrategy",
    "HPUPrecision",
    "HPUCheckpointIO",
]