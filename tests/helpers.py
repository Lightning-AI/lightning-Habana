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
from typing import Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy


class SklearnDataset(Dataset):
    def __init__(self, x, y, x_type, y_type):
        self.x = x
        self.y = y
        self._x_type = x_type
        self._y_type = y_type

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """Get single item."""
        return torch.tensor(self.x[idx], dtype=self._x_type), torch.tensor(self.y[idx], dtype=self._y_type)

    def __len__(self) -> int:
        """Get length."""
        return len(self.y)


class SklearnDataModule(LightningDataModule):
    def __init__(self, sklearn_dataset, x_type, y_type, batch_size: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self._x, self._y = sklearn_dataset
        self._split_data()
        self._x_type = x_type
        self._y_type = y_type

    def _split_data(self):
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._x, self._y, test_size=0.20, random_state=42
        )
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_train, self.y_train, test_size=0.40, random_state=42
        )

    def train_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_train, self.y_train, self._x_type, self._y_type),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_valid, self.y_valid, self._x_type, self._y_type), batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size
        )

    @property
    def sample(self):
        return torch.tensor([self._x[0]], dtype=self._x_type)


class ClassifDataModule(SklearnDataModule):
    def __init__(
        self, num_features=32, length=800, num_classes=3, batch_size=10, n_clusters_per_class=1, n_informative=2
    ):
        from sklearn.datasets import make_classification

        data = make_classification(
            n_samples=length,
            n_features=num_features,
            n_classes=num_classes,
            n_clusters_per_class=n_clusters_per_class,
            n_informative=n_informative,
            random_state=42,
        )
        super().__init__(data, x_type=torch.float32, y_type=torch.long, batch_size=batch_size)


class ClassificationModel(LightningModule):
    def __init__(self, num_features=32, num_classes=3, lr=0.01):
        super().__init__()

        self.lr = lr
        for i in range(3):
            setattr(self, f"layer_{i}", nn.Linear(num_features, num_features))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(num_features, 3))

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)
