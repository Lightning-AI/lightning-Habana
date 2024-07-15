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
from typing import Any, Generator, Mapping, Optional, Union

import torch
from lightning_utilities import module_available
from typing_extensions import get_args

if module_available("lightning"):
    from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_habana.fabric.plugins.precision import _PRECISION_INPUT, HPUPrecision
from lightning_habana.utils.imports import _HPU_SYNAPSE_GREATER_EQUAL_1_14_0
from lightning_habana.utils.resources import _HABANA_FRAMEWORK_AVAILABLE

if _HPU_SYNAPSE_GREATER_EQUAL_1_14_0 and _HABANA_FRAMEWORK_AVAILABLE:
    # Required for training in fp8 using habana transformer engine
    from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling


class HPUFSDPPrecision(FSDPPrecision, HPUPrecision):
    """Plugin that enables mixed precision support on HPUs.

    Args:
        precision: to enable full precision (``'32-true'``), half precision (``'bf16'``) or
            mixed precision (``'bf16-mixed``').
        device: The device for ``torch.autocast``.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT,
        device: str = "hpu",
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        replace_layers: bool = False,
    ) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r}` is not supported." f" `precision` must be one of: {supported_precision}."
            )
        super().__init__(precision)

    def autocast_context_manager(self) -> torch.autocast:
        """Return Autocast context manager."""
        return torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield
