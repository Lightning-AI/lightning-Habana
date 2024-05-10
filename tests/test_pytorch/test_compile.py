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


import pytest
import torch
import torch.nn as nn
import torch.nn.functional as func
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
    from lightning.pytorch.utilities.compile import from_compiled, to_uncompiled
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
from lightning_habana.pytorch.strategies import HPUDDPStrategy, SingleHPUStrategy


@pytest.fixture()
def _is_compile_allowed():
    if HPUAccelerator.is_lazy():
        pytest.skip("Test requires lazy mode to be disabled")


@pytest.mark.usefixtures("_is_compile_allowed")
def test_compiler_context(tmp_path):
    model = BoringModel()
    compiled_model = torch.compile(model, backend="hpu_backend")
    assert model._compiler_ctx is compiled_model._compiler_ctx  # shared reference


@pytest.mark.skipif(not module_available("lightning"), reason="Test requires lightning package")
@pytest.mark.usefixtures("_is_compile_allowed")
def test_lightning_compile_uncompile():
    model = BoringModel()
    compiled_model = torch.compile(model, backend="hpu_backend")

    def has_dynamo(fn):
        return any(el for el in dir(fn) if el.startswith("_torchdynamo"))

    from_compiled_model = from_compiled(compiled_model)
    assert isinstance(from_compiled_model, LightningModule)
    assert from_compiled_model._compiler_ctx is not None
    assert has_dynamo(from_compiled_model.forward)
    assert has_dynamo(from_compiled_model.training_step)
    assert has_dynamo(from_compiled_model.validation_step)
    assert has_dynamo(from_compiled_model.test_step)
    assert has_dynamo(from_compiled_model.predict_step)

    to_uncompiled_model = to_uncompiled(model)
    assert to_uncompiled_model._compiler_ctx is None
    assert to_uncompiled_model.forward == model.forward
    assert to_uncompiled_model.training_step == model.training_step
    assert to_uncompiled_model.validation_step == model.validation_step
    assert to_uncompiled_model.test_step == model.test_step
    assert to_uncompiled_model.predict_step == model.predict_step
    assert not has_dynamo(to_uncompiled_model.forward)
    assert not has_dynamo(to_uncompiled_model.training_step)
    assert not has_dynamo(to_uncompiled_model.validation_step)
    assert not has_dynamo(to_uncompiled_model.test_step)
    assert not has_dynamo(to_uncompiled_model.predict_step)


@pytest.mark.usefixtures("_is_compile_allowed")
def test_compiled_model_to_log_metric(tmp_path):
    class MyModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("loss", loss)
            return loss

    model = MyModel()
    compiled_model = torch.compile(model, backend="hpu_backend")

    _strategy = SingleHPUStrategy()

    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator=HPUAccelerator(),
        fast_dev_run=True,
        strategy=_strategy,
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(compiled_model)

    assert set(trainer.callback_metrics) == {"loss"}


class LitClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = func.cross_entropy(self(x), y)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


@pytest.mark.usefixtures("_is_compile_allowed")
def test_compiled_model_with_datamodule_and_log_metric(tmp_path):
    dm = MNISTDataModule(batch_size=32)
    model = LitClassifier()
    compiled_model = torch.compile(model, backend="hpu_backend")
    _strategy = SingleHPUStrategy()

    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator=HPUAccelerator(),
        fast_dev_run=True,
        strategy=_strategy,
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(compiled_model, datamodule=dm)


@pytest.mark.usefixtures("_is_compile_allowed")
def test_trainer_fit_with_compiled_model(tmp_path):
    """Tests compiled BoringModel on HPU."""
    model = BoringModel()
    compiled_model = torch.compile(model, backend="hpu_backend")

    _strategy = SingleHPUStrategy()
    _plugins = [HPUPrecisionPlugin(precision="bf16-mixed")]

    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator=HPUAccelerator(),
        strategy=_strategy,
        plugins=_plugins,
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(compiled_model)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return func.log_softmax(x, dim=1)


@pytest.mark.usefixtures("_is_compile_allowed")
def test_trainer_with_nn_module(tmp_path):
    device = torch.device("hpu")
    model = Net().to(device)
    torch.compile(model, backend="hpu_backend")


@pytest.mark.parametrize("hpus", [1])
@pytest.mark.usefixtures("_is_compile_allowed")
def test_all_stages_with_compile(tmpdir, hpus):
    """Tests all the model stages using BoringModel on HPU."""
    model_to_train = BoringModel()
    model_to_eval = BoringModel()
    compiled_train_model = torch.compile(model_to_train, backend="hpu_backend")
    compiled_eval_model = torch.compile(model_to_eval, backend="hpu_backend")

    _strategy = SingleHPUStrategy()
    _plugins = [HPUPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        strategy=_strategy,
        devices=hpus,
        plugins=_plugins,
    )
    trainer.fit(compiled_train_model)
    trainer.validate(compiled_eval_model)
    trainer.test(compiled_eval_model)
    trainer.predict(compiled_eval_model)


@pytest.mark.standalone()
@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
@pytest.mark.usefixtures("_is_compile_allowed")
def test_ddp_strategy_with_compile(tmp_path, hpus):
    """Tests compiled BoringModel on HPU."""
    model = BoringModel()
    compiled_model = torch.compile(model, backend="hpu_backend")

    _plugins = [HPUPrecisionPlugin(precision="bf16-mixed")]
    parallel_hpus = [torch.device("hpu")] * hpus
    _strategy = HPUDDPStrategy(
        parallel_devices=parallel_hpus,
        bucket_cap_mb=100,
        gradient_as_bucket_view=True,
        static_graph=True,
        find_unused_parameters=True,
    )

    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator=HPUAccelerator(),
        strategy=_strategy,
        plugins=_plugins,
        devices=hpus,
        fast_dev_run=True,
    )
    trainer.fit(compiled_model)
    assert _strategy._ddp_kwargs["bucket_cap_mb"] == 100
    assert _strategy._ddp_kwargs["gradient_as_bucket_view"] is True
    assert _strategy._ddp_kwargs["static_graph"] is True
    assert _strategy._ddp_kwargs["find_unused_parameters"] is True
