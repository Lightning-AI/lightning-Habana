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

if module_available("lightning"):
    from lightning.pytorch import Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy


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
        fast_dev_run=True, default_root_dir=tmpdir, strategy=HPUDeepSpeedStrategy()
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
        accelerator="hpu",
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


# TBD - deepspeed hpu 1.8 is used, need to be move to 1.9
def test_warn_hpu_deepspeed_ignored(tmpdir):
    class TestModel(BoringModel):
        def backward(self, loss: Tensor, *args, **kwargs) -> None:
            return loss.backward()

    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    model = TestModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        strategy=HPUDeepSpeedStrategy(),
        plugins=_plugins,
        accelerator="hpu",
        devices=1,
        precision="bf16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.warns(UserWarning, match="will be ignored since DeepSpeed handles the backward"):
        trainer.fit(model)
