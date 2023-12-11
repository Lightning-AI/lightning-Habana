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

import torch
from lightning_habana.fabric.accelerator import HPUAccelerator
from lightning_habana.fabric.strategies.parallel import HPUParallelStrategy
from lightning_habana.fabric.strategies.single import SingleHPUStrategy


def test_single_device_default_device():
    assert SingleHPUStrategy().root_device == torch.device("hpu")


def test_hpu_parallel_strategy_defaults():
    strategy = HPUParallelStrategy()
    assert strategy.process_group_backend == "hccl"
    assert len(strategy.parallel_devices) == HPUAccelerator.auto_device_count()
