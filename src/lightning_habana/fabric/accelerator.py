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

from typing import Dict, List, Optional, Union

import torch
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch.accelerators.accelerator import Accelerator
elif module_available("pytorch_lightning"):
    from pytorch_lightning.accelerators.accelerator import Accelerator
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_habana import HPU_AVAILABLE
from lightning_habana.utils.resources import _parse_hpus, device_count

if HPU_AVAILABLE:
    import habana_frameworks.torch.hpu as torch_hpu


class HPUAccelerator(Accelerator):
    """Accelerator for HPU devices."""

    def setup_device(self, device: torch.device) -> None:
        """Setup hpu accelerator.

        Raises:
            ValueError:
                If the selected device is not HPU.
        """
        if device.type != "hpu":
            raise ValueError(f"Device should be HPU, got {device} instead.")

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[int]:
        """Accelerator device parsing logic."""
        return _parse_hpus(devices)

    @staticmethod
    def get_parallel_devices(devices: int) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("hpu")] * devices

    @staticmethod
    def auto_device_count() -> int:
        """Returns the number of HPU devices when the devices is set to auto."""
        return device_count()

    @staticmethod
    def is_available() -> bool:
        """Returns a bool indicating if HPU is currently available."""
        try:
            return torch_hpu.is_available()
        except (AttributeError, NameError):
            return False

    @staticmethod
    def get_device_name() -> str:
        """Returns the name of the HPU device."""
        try:
            return torch_hpu.get_device_name()
        except (AttributeError, NameError):
            return ""

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "hpu",
            cls,
            description=cls.__class__.__name__,
        )
