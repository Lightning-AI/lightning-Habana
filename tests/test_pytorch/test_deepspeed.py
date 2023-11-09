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
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy
from lightning_habana.pytorch.strategies.deepspeed import _HPU_DEEPSPEED_AVAILABLE

if _HPU_DEEPSPEED_AVAILABLE:
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
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
def get_device_count(pytestconfig):
    hpus = int(pytestconfig.getoption("hpus"))
    if not hpus:
        assert HPUAccelerator.auto_device_count() >= 1
        return 1
    assert hpus <= HPUAccelerator.auto_device_count(), "More hpu devices asked than present"
    return hpus


@pytest.fixture()
def deepspeed_base_config():
    return {
        "train_batch_size": 8,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "train_micro_batch_size_per_gpu": 2,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0.02,
                "warmup_max_lr": 0.05,
                "warmup_num_steps": 4,
                "total_num_steps": 8,
                "warmup_type": "linear",
            },
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {"stage": 0},
    }


def config_generator(
    deepspeed_base_config,
    stage,
    cpu_offload,
    activation_checkpoints,
    partition_activations,
    contiguous_checkpointing,
    checkpoint_in_cpu,
):
    deepspeed_config = {**deepspeed_base_config}

    deepspeed_config["zero_optimization"]["stage"] = stage
    if stage == "infinity":
        deepspeed_config["zero_optimization"]["stage"] = 3
        deepspeed_config["zero_optimization"]["offload_param"] = {"device": "cpu"}

    if cpu_offload:
        deepspeed_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        deepspeed_config["zero_optimization"]["contiguous_gradients"] = True
        deepspeed_config["zero_optimization"]["overlap_comm"] = True

    if stage != 0 and activation_checkpoints:
        deepspeed_config["activation_checkpointing"] = {
            "partition_activations": partition_activations,
            "contiguous_memory_optimization": False,
            "cpu_checkpointing": checkpoint_in_cpu,
        }

    if stage == 0:
        deepspeed_config["bf16"]["enabled"] = False

    return deepspeed_config


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


@pytest.fixture()
def deepspeed_zero_autotuning_config():
    return {
        "bf16": {"enabled": True},
        "autotuning": {
            "enabled": True,
            "arg_mappings": {
                "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
                "gradient_accumulation_steps ": "--gradient_accumulation_steps",
            },
        },
    }


@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
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


class SampleDataset(Dataset):
    def __init__(self, batch_size, data_size):
        x = torch.ones([batch_size, data_size], dtype=torch.float, device="hpu")
        y = torch.zeros([batch_size, data_size], dtype=torch.float, device="hpu")
        self.train_data = (x, y)

    def __getitem__(self, index):
        """Get a sample."""
        return (self.train_data[0][index], self.train_data[1][index])

    def __len__(self):
        """Get length of dataset."""
        return self.train_data[0].size(0)


class SampleLayer(torch.nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones([data_size], dtype=torch.float))

    def forward(self, input):
        return input * self.w


class SampleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = SampleLayer(10)
        self.l2 = SampleLayer(10)
        self.l3 = SampleLayer(10)
        self.l4 = SampleLayer(10)

    def forward(self, x):
        l1_out = self.l1(x)
        l2_out = checkpoint(self.l2, l1_out)
        l3_out = checkpoint(self.l3, l2_out)
        return checkpoint(self.l4, l3_out)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.sum(torch.abs(y - logits)) / (2 * 10 * 50)
        self.log("train_loss", loss.item(), sync_dist=True, reduce_fx="sum")
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.sum(torch.abs(y - logits)) / (2 * 10 * 50)
        self.log("valid_loss", loss, sync_dist=True, reduce_fx="sum")
        return {"loss": loss, "logits": logits}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.sum(torch.abs(y - logits)) / (2 * 10 * 50)
        self.log("test_loss", loss, sync_dist=True, reduce_fx="sum")
        return {"loss": loss, "logits": logits}

    def configure_optimizers(self):
        from torch.optim.adamw import AdamW as AdamW

        return torch.optim.AdamW(self.parameters())

    def train_dataloader(self):
        return DataLoader(SampleDataset(16, 10), batch_size=2)

    def val_dataloader(self):
        return DataLoader(SampleDataset(16, 10), batch_size=2)

    def test_dataloader(self):
        return DataLoader(SampleDataset(16, 10), batch_size=2)


def test_deepspeed_config(tmpdir):
    """Test to ensure deepspeed config works correctly.

    DeepSpeed config object including optimizers/schedulers and saves the model weights to load correctly.

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


def test_multi_optimizer_with_hpu_deepspeed(tmpdir):
    """Test to validate multi optimizer support with deepspeed."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def configure_optimizers(self):
            optimizer1 = torch.optim.AdamW(self.parameters())
            optimizer2 = torch.optim.AdamW(self.parameters())
            return [optimizer1, optimizer2]

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
    with pytest.raises(
        MisconfigurationException, match="DeepSpeed currently only supports single optimizer, single optional scheduler"
    ):
        trainer.fit(model)


@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize("zero_config", [0, 1, 2, 3, "infinity"])
@pytest.mark.parametrize("cpu_offload", [True, False])
@pytest.mark.parametrize(
    ("activation_checkpoints", "partition_activations", "contiguous_checkpointing", "checkpoint_in_cpu"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
        (True, False, False, True),
    ],
)
def test_lightning_model(
    deepspeed_base_config,
    zero_config,
    cpu_offload,
    activation_checkpoints,
    partition_activations,
    contiguous_checkpointing,
    checkpoint_in_cpu,
    get_device_count,
):
    """Test that DeepSpeed works with a simple LightningModule and LightningDataModule."""
    config = config_generator(
        deepspeed_base_config,
        zero_config,
        cpu_offload,
        activation_checkpoints,
        partition_activations,
        contiguous_checkpointing,
        checkpoint_in_cpu,
    )

    if zero_config == 3 and HPUAccelerator.get_device_name() == "GAUDI2":
        pytest.skip(reason="Not supported ")

    if zero_config == 0 and cpu_offload is True:
        pytest.skip("Zero stage 0 and cpu_offload is an invalid configuration")

    if zero_config == "infinity" and cpu_offload is False:
        pytest.skip("Not running zero_infinity without cpu_offload")

    model = SampleModel()
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    _accumulate_grad_batches = config["train_micro_batch_size_per_gpu"]
    _batch_size = 2
    _parallel_hpus = [torch.device("hpu")] * get_device_count

    config["train_batch_size"] = get_device_count * _accumulate_grad_batches * _batch_size

    trainer = Trainer(
        accelerator=HPUAccelerator(),
        strategy=HPUDeepSpeedStrategy(config=config, parallel_devices=_parallel_hpus),
        enable_progress_bar=False,
        fast_dev_run=8,
        plugins=_plugins,
        use_distributed_sampler=False,
        limit_train_batches=16,
        accumulate_grad_batches=_accumulate_grad_batches,
    )

    trainer.fit(model)
    expected = torch.tensor([0.0164])
    current_loss = trainer.callback_metrics["train_loss"].detach().to("cpu")
    assert torch.allclose(
        current_loss, expected, atol=4e-4
    ), f"incorrect loss value {current_loss}, expected {expected}"


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("offload", [True, False])
def test_lightning_deepspeed_stages(get_device_count, zero_stage, offload):
    model = SampleModel()
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=get_device_count,
        strategy=HPUDeepSpeedStrategy(zero_optimization=True, stage=zero_stage, offload_optimizer=offload),
        plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")],
        fast_dev_run=8,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        limit_train_batches=16,
        accumulate_grad_batches=2,
    )
    trainer.fit(model)


def test_hpu_deepspeed_with_invalid_optimizer():
    """Test to ensure if we pass an invalid optimizer and throws an exception."""

    class DummyModel(BoringModel):
        def configure_optimizers(self):
            return None

    import logging

    model = DummyModel()
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        strategy=HPUDeepSpeedStrategy(logging_level=logging.INFO),
        max_epochs=1,
        plugins=_plugins,
        devices=1,
    )
    with pytest.raises(
        MisconfigurationException, match="You have specified an invalid optimizer to be run with deepspeed."
    ):
        trainer.fit(model)


def test_hpu_deepspeed_with_optimizer_and_config(deepspeed_zero_config):
    """Test the preference of optimizer when configured both from deepspeed config and LightningModule."""

    class DummyModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.1)

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            from deepspeed.runtime.lr_schedules import WarmupLR

            assert isinstance(trainer.optimizers[0], DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.AdamW)
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, WarmupLR)
            assert trainer.lr_scheduler_configs[0].interval == "step"

    import logging

    model = DummyModel()

    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        strategy=HPUDeepSpeedStrategy(logging_level=logging.INFO, config=deepspeed_zero_config),
        callbacks=[TestCB()],
        max_epochs=1,
        plugins=_plugins,
        devices=1,
    )
    trainer.fit(model)


@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
def test_deepspeed_resume_training(tmpdir, deepspeed_base_config, get_device_count):
    """Test to ensure with Stage 3 and single GPU that we can resume training."""
    initial_model = SampleModel()
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    _zero_stage = 3
    config = config_generator(
        deepspeed_base_config,
        _zero_stage,
        True,
        True,
        False,
        False,
        True,
    )
    _accumulate_grad_batches = config["train_micro_batch_size_per_gpu"]
    _batch_size = 2
    _parallel_hpus = [torch.device("hpu")] * get_device_count

    config["train_batch_size"] = get_device_count * _accumulate_grad_batches * _batch_size
    config_copy = config.copy()
    ck = ModelCheckpoint(monitor="train_loss", mode="max", save_last=True, save_top_k=-1)
    initial_trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=HPUDeepSpeedStrategy(config=config, parallel_devices=_parallel_hpus),
        accelerator=HPUAccelerator(),
        accumulate_grad_batches=_accumulate_grad_batches,
        plugins=_plugins,
        callbacks=[ck],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    initial_trainer.fit(initial_model)

    class TestCallback(Callback):
        def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            original_deepspeed_strategy = initial_trainer.strategy
            current_deepspeed_strategy = trainer.strategy

            assert isinstance(original_deepspeed_strategy, HPUDeepSpeedStrategy)
            assert isinstance(current_deepspeed_strategy, HPUDeepSpeedStrategy)
            # assert optimizer states are the correctly loaded
            original_optimizer_dict = original_deepspeed_strategy.deepspeed_engine.optimizer.state_dict()
            current_optimizer_dict = current_deepspeed_strategy.deepspeed_engine.optimizer.state_dict()
            for orig_tensor, current_tensor in zip(
                original_optimizer_dict["fp32_flat_groups"], current_optimizer_dict["fp32_flat_groups"]
            ):
                assert torch.all(orig_tensor.eq(current_tensor))
            # assert model state is loaded correctly
            for current_param, initial_param in zip(pl_module.parameters(), initial_model.parameters()):
                assert torch.equal(current_param.cpu(), initial_param.cpu())
            # assert epoch has correctly been restored
            assert trainer.current_epoch == 1

            # assert lr-scheduler states are loaded correctly
            original_lr_scheduler = initial_trainer.lr_scheduler_configs[0].scheduler
            current_lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert original_lr_scheduler.state_dict() == current_lr_scheduler.state_dict()

    model = SampleModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=HPUDeepSpeedStrategy(config=config_copy, parallel_devices=_parallel_hpus),
        accelerator=HPUAccelerator(),
        accumulate_grad_batches=_accumulate_grad_batches,
        plugins=_plugins,
        callbacks=TestCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, ckpt_path=ck.best_model_path)


class TestLayer(torch.nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones([data_size]))

    def forward(self, input):
        return input * torch.matmul(input, self.w)


class InferenceModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = TestLayer(2)
        self.l2 = TestLayer(2)
        self.l3 = TestLayer(2)
        self.l4 = TestLayer(2)

    def forward(self, x):
        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        return self.l4(l3_out)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def predict_dataloader(self):
        return DataLoader(SampleDataset(1, 2))


@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_lightning_deepspeed_inference_kwargs(enable_cuda_graph, get_device_count):
    model = InferenceModel()
    kwargs = {"dtype": torch.float}
    kwargs["tensor_parallel"] = {"tp_size": get_device_count}
    kwargs["enable_cuda_graph"] = enable_cuda_graph
    kwargs["replace_method"] = "auto"
    kwargs["replace_with_kernel_inject"] = False
    kwargs["injection_policy"] = {InferenceModel: ("l1")}
    _parallel_hpus = [torch.device("hpu")] * get_device_count

    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=get_device_count,
        strategy=HPUDeepSpeedStrategy(parallel_devices=_parallel_hpus, **kwargs),
        plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")],
        use_distributed_sampler=False,
    )
    preds = trainer.predict(model)
    expected = torch.tensor([32768.0, 32768.0])
    assert torch.allclose(preds[0], expected), f"incorrect result value {preds}, expected {expected}"


@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
def test_lightning_deepspeed_inference_params(get_device_count, dtype):
    if dtype == torch.float16 and HPUAccelerator.get_device_name() == "GAUDI":
        pytest.skip(reason="FP16 is not supported by Gaudi1")

    model = InferenceModel()
    _parallel_hpus = [torch.device("hpu")] * get_device_count

    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=get_device_count,
        strategy=HPUDeepSpeedStrategy(
            parallel_devices=_parallel_hpus,
            tensor_parallel={"tp_size": get_device_count},
            dtype=dtype,
            replace_with_kernel_inject=True,
        ),
        plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")],
        use_distributed_sampler=False,
    )
    preds = trainer.predict(model)
    expected = torch.tensor([32768.0, 32768.0])
    assert torch.allclose(preds[0], expected), f"incorrect result value {preds}, expected {expected}"


@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
def test_lightning_deepspeed_inference_config(get_device_count, dtype):
    if dtype == torch.float16 and HPUAccelerator.get_device_name() == "GAUDI":
        pytest.skip(reason="FP16 is not supported by Gaudi1")

    model = InferenceModel()
    _parallel_hpus = [torch.device("hpu")] * get_device_count

    _config = {
        "replace_with_kernel_inject": True,
        "tensor_parallel": {"tp_size": get_device_count},
        "dtype": dtype,
        "enable_cuda_graph": False,
    }

    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=get_device_count,
        strategy=HPUDeepSpeedStrategy(
            parallel_devices=_parallel_hpus,
            config=_config,
        ),
        plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")],
        use_distributed_sampler=False,
    )
    preds = trainer.predict(model)
    expected = torch.tensor([32768.0, 32768.0])
    assert torch.allclose(preds[0], expected), f"incorrect result value {preds}, expected {expected}"
