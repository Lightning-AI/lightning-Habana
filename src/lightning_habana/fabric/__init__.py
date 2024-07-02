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


from lightning_habana.fabric.accelerator import HPUAccelerator
from lightning_habana.fabric.plugins.fsdp_precision import HPUFSDPPrecision
from lightning_habana.fabric.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.fabric.plugins.precision import HPUPrecision
from lightning_habana.fabric.strategies.ddp import HPUDDPStrategy
from lightning_habana.fabric.strategies.fsdp import HPUFSDPStrategy
from lightning_habana.fabric.strategies.parallel import HPUParallelStrategy
from lightning_habana.fabric.strategies.single import SingleHPUStrategy

__all__ = [
    "HPUAccelerator",
    "HPUDDPStrategy",
    "HPUParallelStrategy",
    "SingleHPUStrategy",
    "HPUPrecision",
    "HPUCheckpointIO",
    "HPUFSDPStrategy",
    "HPUFSDPPrecision",
]
