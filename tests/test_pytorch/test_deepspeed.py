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

import json
import os
from typing import Any, Dict

import pytest
import torch
from lightning_utilities import module_available
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, LightningModule
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy
from lightning_habana.pytorch.strategies.deepspeed import _HPU_DEEPSPEED_AVAILABLE

if _HPU_DEEPSPEED_AVAILABLE:
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


class ModelParallelBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.configure_sharded_model()


class ModelParallelBoringModelNoSchedulers(ModelParallelBoringModel):
    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class ModelParallelBoringModelManualOptim(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.configure_sharded_model()

    @property
    def automatic_optimization(self) -> bool:
        return False


@pytest.fixture()
def deepspeed_config():
    return {
        "optimizer": {"type": "SGD", "params": {"lr": 3e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"last_batch_iteration": -1, "warmup_min_lr": 0, "warmup_max_lr": 3e-5, "warmup_num_steps": 100},
        },
    }


@pytest.fixture()
def deepspeed_zero_config(deepspeed_config):
    return {**deepspeed_config, "zero_allow_untested_optimizer": True, "zero_optimization": {"stage": 2}}


def test_hpu_deepspeed_strategy_env(tmpdir, monkeypatch, deepspeed_config):
    """Test to ensure that the strategy can be passed via a string with an environment variable."""
    config_path = os.path.join(tmpdir, "temp.json")
    with open(config_path, "w") as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)

    trainer = Trainer(
        accelerator=HPUAccelerator(), fast_dev_run=True, default_root_dir=tmpdir, strategy=HPUDeepSpeedStrategy()
    )  # strategy="hpu_deepspeed")

    strategy = trainer.strategy
    assert isinstance(strategy, HPUDeepSpeedStrategy)
    assert len(trainer.strategy.parallel_devices) > 1
    assert trainer.strategy.parallel_devices[0] == torch.device("hpu")
    assert strategy.config == deepspeed_config


def test_hpu_deepspeed_precision_choice(tmpdir):
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=HPUDeepSpeedStrategy(),  # strategy="hpu_deepspeed",
        plugins=_plugins,
        precision="bf16-mixed",
    )

    assert isinstance(trainer.strategy, HPUDeepSpeedStrategy)
    assert isinstance(trainer.strategy.precision_plugin, DeepSpeedPrecisionPlugin)
    assert trainer.strategy.precision_plugin.precision == "bf16-mixed"


def test_hpu_deepspeed_with_invalid_config_path():
    """Test to ensure if we pass an invalid config path we throw an exception."""
    with pytest.raises(
        MisconfigurationException, match="You passed in a path to a DeepSpeed config but the path does not exist"
    ):
        HPUDeepSpeedStrategy(config="invalid_path.json")


def test_deepspeed_defaults():
    """Ensure that defaults are correctly set as a config for DeepSpeed if no arguments are passed."""
    strategy = HPUDeepSpeedStrategy()
    assert strategy.config is not None
    assert isinstance(strategy.config["zero_optimization"], dict)


def test_warn_hpu_deepspeed_ignored(tmpdir):
    class TestModel(BoringModel):
        def backward(self, loss: Tensor, *args, **kwargs) -> None:
            return loss.backward()

    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    model = TestModel()
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        fast_dev_run=True,
        default_root_dir=tmpdir,
        strategy=HPUDeepSpeedStrategy(),
        plugins=_plugins,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.warns(UserWarning, match="will be ignored since DeepSpeed handles the backward"):
        trainer.fit(model)


def test_deepspeed_config(tmpdir):
    """Test to ensure deepspeed config works correctly.

    DeepSpeed config object including
    optimizers/schedulers and saves the model weights to load correctly.
    """

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            from torch.optim.lr_scheduler import StepLR

            assert isinstance(trainer.optimizers[0], DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, StepLR)
            assert trainer.lr_scheduler_configs[0].interval == "epoch"

    model = BoringModel()
    lr_monitor = LearningRateMonitor()
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        strategy=HPUDeepSpeedStrategy(),
        default_root_dir=tmpdir,
        devices=1,
        log_every_n_steps=1,
        limit_train_batches=4,
        limit_val_batches=4,
        limit_test_batches=4,
        max_epochs=2,
        plugins=_plugins,
        callbacks=[TestCB(), lr_monitor],
        logger=CSVLogger(tmpdir),
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model)
    trainer.test(model)
    assert list(lr_monitor.lrs) == ["lr-SGD"]
    assert len(set(lr_monitor.lrs["lr-SGD"])) == trainer.max_epochs


class SomeDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        """Get a sample."""
        return self.data[index]

    def __len__(self):
        """Get length of dataset."""
        return self.len


class SomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(SomeDataset(32, 64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(SomeDataset(32, 64), batch_size=2)


def test_lightning_model():
    """Test that DeepSpeed works with a simple LightningModule and LightningDataModule."""
    model = SomeModel()
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(), max_epochs=1, plugins=_plugins, devices=1
    )
    trainer.fit(model)
