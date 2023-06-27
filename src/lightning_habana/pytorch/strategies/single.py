# Copyright The Lightning AI team.
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

from typing import Any, Callable, Dict, Optional, Union

from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.plugins import CheckpointIO
    from lightning.fabric.utilities.types import _DEVICE
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.utilities.types import _DEVICE
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.core as htcore


class SingleHPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single HPU device."""

    strategy_name = "hpu_single"

    def __init__(
        self,
        device: _DEVICE = "hpu",
        accelerator: Optional[Accelerator] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:  # type: ignore[has-type]
            self._checkpoint_io = HPUCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = HPUCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io  # type: ignore

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: Trainer) -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_optimizers(self, trainer: Trainer) -> None:
        super().setup_optimizers(trainer)

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def on_after_backward(self) -> None:
        # Break lazy accumulation of graph after fwd+bwd
        htcore.mark_step()

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Callable[[], Any],
        model: Optional[Union[LightningModule, Module]] = None,
        **kwargs: Any,
    ) -> Any:
        optimizer_output = super().optimizer_step(optimizer, closure, model, **kwargs)
        # Break lazy accumulation of graph after optimizer
        htcore.mark_step()
        return optimizer_output

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().predict_step(batch, batch_idx)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
