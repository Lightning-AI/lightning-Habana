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

import pytest
import torch

from lightning_habana.fabric.accelerator.accelerator import HPUAccelerator


def test_auto_device_count():
    assert HPUAccelerator.auto_device_count() != 0


def test_availability():
    assert HPUAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be HPU"):
        HPUAccelerator().setup_device(torch.device("cpu"))


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        (1, [torch.device("hpu")]),
        (2, [torch.device("hpu")] * 2),
        (8, [torch.device("hpu")] * 8),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert HPUAccelerator.get_parallel_devices(devices) == expected
