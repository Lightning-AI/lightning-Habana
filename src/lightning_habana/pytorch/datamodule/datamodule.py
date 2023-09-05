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

import warnings
from typing import Any, Optional

from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningDataModule
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningDataModule

import torch

from lightning_habana.utils.imports import (
    _HABANA_FRAMEWORK_AVAILABLE,
    _LIGHTNING_GREATER_EQUAL_2_0_0,
    _TORCH_GREATER_EQUAL_2_0_0,
    _TORCHVISION_AVAILABLE,
)

if _TORCHVISION_AVAILABLE:
    import torchvision.datasets
    from torchvision import transforms as transform_lib

if _HABANA_FRAMEWORK_AVAILABLE:
    try:
        import habana_dataloader
    except ImportError:
        raise ModuleNotFoundError("habana_dataloader package is not installed.")

    from lightning_habana.pytorch.datamodule.dataloaders.resnet_media_pipe import MediaApiDataLoader

import lightning_habana.pytorch.datamodule.utils

_DATASETS_PATH = "/tmp/data"


def patch_aeon_length(self) -> int:  # type: ignore[no-untyped-def]
    """WA to avoid hang in aeon dataloader with PyTorch Lightning version >= 2.0.0.

    Returns adjusted length if lightning version >= 2.0.0
    Returns default length otherwise.
    """
    length = len(self.dataloader)
    # If Lightning version is >= 2.0.0, and dataloader is aeon,
    # drop the last batch from dataloader.
    if _LIGHTNING_GREATER_EQUAL_2_0_0 and _TORCH_GREATER_EQUAL_2_0_0 and self.dataloader.aeon is not None:
        return length - 1
    return length


def load_data(traindir, valdir, train_transforms, val_transforms):  # type: ignore[no-untyped-def]
    """Helper to initialize dataset and transforms."""
    # check supported transforms
    if train_transforms is not None:
        for t in train_transforms:
            if isinstance(
                t,
                (
                    transform_lib.RandomResizedCrop,
                    transform_lib.RandomHorizontalFlip,
                    transform_lib.ToTensor,
                    transform_lib.Normalize,
                ),
            ):
                print("Train Transforms supported")
            else:
                raise ValueError("Unsupported train transform: " + str(type(t)))

        train_transforms = transform_lib.Compose(train_transforms)

    if val_transforms is not None:
        for t in val_transforms:
            if isinstance(
                t, (transform_lib.Resize, transform_lib.CenterCrop, transform_lib.ToTensor, transform_lib.Normalize)
            ):
                print("Val Transforms supported")
            else:
                raise ValueError("Unsupported val transform: " + str(type(t)))

        val_transforms = transform_lib.Compose(val_transforms)

    if "imagenet" not in traindir.lower() and "ilsvrc2012" not in traindir.lower():
        raise ValueError("Habana dataloader only supports Imagenet dataset")

    # Data loading code
    dataset_train = torchvision.datasets.ImageFolder(traindir, train_transforms)

    dataset_val = torchvision.datasets.ImageFolder(valdir, val_transforms)

    return dataset_train, dataset_val


class HPUDataModule(LightningDataModule):
    """Datamodule helper class to load the right media pipe."""

    name = "hpu-dataset"

    def __init__(
        self,
        train_dir: str = _DATASETS_PATH,
        val_dir: str = _DATASETS_PATH,
        num_workers: int = 8,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        train_transforms: Any = None,
        val_transforms: Any = None,
        pin_memory: bool = True,
        shuffle: bool = False,
        drop_last: bool = True,
        distributed: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.distributed = distributed
        self.data_loader_type = torch.utils.data.DataLoader

        if _HABANA_FRAMEWORK_AVAILABLE:
            if lightning_habana.pytorch.datamodule.utils.is_gaudi2():
                self.data_loader_type = MediaApiDataLoader  # type: ignore
            else:
                self.data_loader_type = habana_dataloader.HabanaDataLoader

    def setup(self, stage: Optional[str] = None):  # type: ignore[no-untyped-def]
        """Method to sanitize the input params."""
        if not _TORCHVISION_AVAILABLE:
            raise ValueError("torchvision transforms not available")

        if self.shuffle is True:
            raise ValueError("HabanaDataLoader does not support shuffle=True")

        if self.pin_memory is False:
            raise ValueError("HabanaDataLoader only supports pin_memory=True")

        if self.num_workers != 8:
            raise ValueError("HabanaDataLoader only supports num_workers as 8")

        self.dataset_train, self.dataset_val = load_data(
            self.train_dir, self.val_dir, self.train_transform, self.val_transform
        )

        dataset = self.dataset_train if stage == "fit" else self.dataset_val
        if self.drop_last is False and (
            isinstance(
                dataset, (torchvision.datasets.ImageFolder, habana_dataloader.habana_dataset.ImageFolderWithManifest)
            )
        ):
            warnings.warn("HabanaDataLoader only supports drop_last as True with Imagenet dataset. Setting to True")
            self.drop_last = True
        if dataset is None:
            raise TypeError("Error creating dataset")

    def train_dataloader(self):  # type: ignore[no-untyped-def]
        """Train set removes a subset to use for validation."""
        sampler_train = (
            torch.utils.data.distributed.DistributedSampler(self.dataset_train)
            if self.distributed
            else torch.utils.data.RandomSampler(self.dataset_train)
        )
        if self.drop_last and lightning_habana.pytorch.datamodule.utils.is_gaudi():
            self.data_loader_type.__len__ = patch_aeon_length
        return self.data_loader_type(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_train,
        )

    def val_dataloader(self):  # type: ignore[no-untyped-def]
        """Val set uses a subset of the training set for validation."""
        sampler_eval = (
            torch.utils.data.distributed.DistributedSampler(self.dataset_val)
            if self.distributed
            else torch.utils.data.SequentialSampler(self.dataset_val)
        )
        if self.drop_last and lightning_habana.pytorch.datamodule.utils.is_gaudi():
            self.data_loader_type.__len__ = patch_aeon_length
        return self.data_loader_type(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_eval,
        )

    def test_dataloader(self):  # type: ignore[no-untyped-def]
        """Test set uses the test split."""
        sampler_eval = (
            torch.utils.data.distributed.DistributedSampler(self.dataset_val)
            if self.distributed
            else torch.utils.data.SequentialSampler(self.dataset_val)
        )
        if self.drop_last and lightning_habana.pytorch.datamodule.utils.is_gaudi():
            self.data_loader_type.__len__ = patch_aeon_length
        return self.data_loader_type(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_eval,
        )
