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
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningDataModule, Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningDataModule, Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

from lightning_habana.pytorch.datamodule.datamodule import HPUDataModule

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms


def test_hpu_datamodule():
    data_module = HPUDataModule(num_workers=2, batch_size=32, shuffle=False, pin_memory=True)
    assert isinstance(data_module, LightningDataModule)


def test_hpu_datamodule_shuffle():
    data_module = HPUDataModule(num_workers=2, batch_size=32, shuffle=True, pin_memory=True)

    model = BoringModel()
    trainer = Trainer(devices=1, accelerator="hpu", max_epochs=1)
    with pytest.raises(ValueError, match="HabanaDataLoader does not support shuffle=True"):
        trainer.fit(model, datamodule=data_module)


def test_hpu_datamodule_pin_memory():
    data_module = HPUDataModule(
        num_workers=2,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
    )

    model = BoringModel()
    trainer = Trainer(devices=1, accelerator="hpu", max_epochs=1)
    with pytest.raises(ValueError, match="HabanaDataLoader only supports pin_memory=True"):
        trainer.fit(model, datamodule=data_module)


def test_hpu_datamodule_num_workers():
    data_module = HPUDataModule(num_workers=4, batch_size=32, shuffle=False, pin_memory=True)

    model = BoringModel()
    trainer = Trainer(devices=1, accelerator="hpu", max_epochs=1)
    with pytest.raises(ValueError, match="HabanaDataLoader only supports num_workers as 2"):
        trainer.fit(model, datamodule=data_module)


def test_hpu_datamodule_unsupported_transforms():
    if not _TORCHVISION_AVAILABLE:
        return

    model = BoringModel()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.ColorJitter()
    train_transforms = [
        transform,
        transforms.ToTensor(),
        normalize,
    ]

    data_module = HPUDataModule(
        train_transforms=train_transforms, num_workers=2, batch_size=32, shuffle=False, pin_memory=True
    )

    # Initialize a trainer
    trainer = Trainer(
        devices=1,
        accelerator="hpu",
        max_epochs=1,
        precision=32,
        max_steps=1,
        limit_test_batches=0.1,
        limit_val_batches=0.1,
    )

    with pytest.raises(ValueError, match=f"Unsupported train transform: {str(type(transform))}"):
        trainer.fit(model, datamodule=data_module)
