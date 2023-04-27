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
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import SingleHPUStrategy


class LitClassifier(LightningModule):
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


if __name__ == "__main__":
    dm = MNISTDataModule(batch_size=32)
    model = LitClassifier()
    trainer = Trainer(fast_dev_run=True, accelerator=HPUAccelerator(), devices=1, strategy=SingleHPUStrategy())

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
