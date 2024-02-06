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


from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Optional, Union

import torch
from lightning_utilities import module_available
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import get_args, override

if module_available("lightning"):
    import lightning.pytorch as pl
    from lightning.fabric.utilities.types import Steppable
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
    from lightning.pytorch.utilities import GradClipAlgorithmType
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
    from lightning.pytorch.utilities.model_helpers import is_overridden
    from lightning.pytorch.utilities.rank_zero import WarningCache
elif module_available("pytorch_lightning"):
    from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

_PRECISION_INPUT = Literal["32", "32-true", "bf16", "bf16-mixed"]

if TYPE_CHECKING:
    import deepspeed

warning_cache = WarningCache()


class HPUDeepSpeedPrecisionPlugin(PrecisionPlugin):
    """Plugin that enables bfloat support on HPUs.

    Args:
        precision: to enable ``torch.bfloat16`` (``'bf16-mixed'``).
        device: The device for ``torch.autocast``.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT,
        device: str = "hpu",
    ) -> None:
        # if not _HPU_SYNAPSE_GREATER_EQUAL_1_11_0:
        #     raise OSError("HPU precision plugin requires `Synapse AI release >= 1.11.0`.")
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(accelerator='hpu', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = precision
        self.device = device

    def autocast_context_manager(self) -> torch.autocast:
        return torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    @override
    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs back-propagation using DeepSpeed's engine.

        Args:
            tensor: the loss tensor
            model: the model to be optimized
            optimizer: ignored for DeepSpeed
            *args: additional positional arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call
            **kwargs: additional keyword arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call

        """
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        deepspeed_engine: "deepspeed.DeepSpeedEngine" = model.trainer.model
        deepspeed_engine.backward(tensor, *args, **kwargs)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException("DeepSpeed and the LBFGS optimizer are not compatible.")
        closure_result = closure()
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `DeepSpeed`"
            )
        # DeepSpeed handles the optimizer step internally
        deepspeed_engine: "deepspeed.DeepSpeedEngine" = model.trainer.model
        return deepspeed_engine.step(**kwargs)

    @override
    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """DeepSpeed handles gradient clipping internally."""
