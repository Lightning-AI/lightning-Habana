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

import torch
from lightning_utilities import module_available
from torch.nn import functional as F  # noqa: N812

if module_available("lightning"):
    from lightning.pytorch import LightningModule
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule


class LitClassifier(LightningModule):
    """Base model."""

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log("test_acc", acc)

    @staticmethod
    def accuracy(logits, y):
        return torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class LitAutocastClassifier(LitClassifier):
    """Base Model with torch.autocast CM."""

    def __init__(self, op_override=False):
        super().__init__()
        self.op_override = op_override

    def forward(self, x):
        if self.op_override:
            self.check_override(x)
        return super().forward(x)

    def check_override(self, x):
        """Checks for op override."""
        identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        y = torch.mm(x, identity)
        z = torch.tan(x)
        assert y.dtype == torch.float32
        assert z.dtype == torch.bfloat16

    def training_step(self, batch, batch_idx):
        """Training step."""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Test step."""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            return super().test_step(batch, batch_idx)
