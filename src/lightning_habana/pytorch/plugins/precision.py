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
from typing import Generator, Literal

import torch
from lightning_utilities import module_available
from typing_extensions import get_args

if module_available("lightning"):
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
elif module_available("pytorch_lightning"):
    from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_habana.utils.imports import _HPU_SYNAPSE_GREATER_EQUAL_1_11_0

_PRECISION_INPUT = Literal["32", "32-true", "bf16", "bf16-mixed"]


class HPUPrecisionPlugin(PrecisionPlugin):
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
        if not _HPU_SYNAPSE_GREATER_EQUAL_1_11_0:
            raise OSError("HPU precision plugin requires `Synapse AI release >= 1.11.0`.")
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
