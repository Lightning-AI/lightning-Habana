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

from typing import Any, Dict, Optional

from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.accelerators import Accelerator
    from lightning.fabric.plugins import CheckpointIO
    from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
    from lightning.fabric.plugins.precision import Precision
    from lightning.fabric.strategies.single_device import SingleDeviceStrategy
    from lightning.fabric.utilities.types import _DEVICE, Optimizable
elif module_available("pytorch_lightning"):
    from lightning_fabric.accelerators import Accelerator
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
    from lightning_fabric.plugins.precision import Precision
    from lightning_fabric.strategies.single_device import SingleDeviceStrategy
    from lightning_fabric.utilities.types import _DEVICE, Optimizable
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from torch import Tensor
from torch.nn import Module

from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE, _TORCH_LESSER_EQUAL_1_13_1

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.core as htcore

from lightning_habana import HPU_AVAILABLE


class SingleHPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single HPU device."""

    strategy_name = "single_hpu"

    def __init__(
        self,
        device: _DEVICE = "hpu",
        accelerator: Optional[Accelerator] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
    ):
        if not HPU_AVAILABLE:
            raise ValueError("`SingleHPUStrategy` requires HPU devices to run")

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:  # type: ignore
            self._checkpoint_io = TorchCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    def backward(self, tensor: Tensor, module: Optional[Module], *args: Any, **kwargs: Any) -> None:
        super().backward(tensor=Tensor, module=module, *args, **kwargs)
        if _TORCH_LESSER_EQUAL_1_13_1:
            # Break lazy accumulation of graph after fwd+bwd
            htcore.mark_step()

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        optimizer_output = super().optimizer_step(optimizer=optimizer, **kwargs)
        if _TORCH_LESSER_EQUAL_1_13_1:
            # Break lazy accumulation of graph after optimizer
            htcore.mark_step()
        return optimizer_output

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
