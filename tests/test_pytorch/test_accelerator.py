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
import copy
import os
from contextlib import nullcontext
from typing import Any, Optional, Union
from unittest import mock

import pytest
import torch
from lightning_habana.utils.hpu_distributed import supported_reduce_ops
from lightning_habana.utils.resources import device_count
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
from lightning_habana.pytorch.strategies import HPUParallelStrategy, SingleHPUStrategy

from tests.helpers import ClassifDataModule, ClassificationModel


def test_availability():
    assert HPUAccelerator.is_available()


@pytest.mark.standalone()
def test_all_stages(tmpdir, hpus):
    """Tests all the model stages using BoringModel on HPU."""
    model = BoringModel()

    _strategy = SingleHPUStrategy()
    if hpus > 1:
        parallel_hpus = [torch.device("hpu")] * hpus
        _strategy = HPUParallelStrategy(parallel_devices=parallel_hpus)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        strategy=_strategy,
        devices=hpus,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


def test_device_name():
    assert "GAUDI" in HPUAccelerator.get_device_name()


def test_accelerator_selected():
    trainer = Trainer(accelerator=HPUAccelerator(), strategy=SingleHPUStrategy())
    assert isinstance(trainer.accelerator, HPUAccelerator)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_optimization(tmpdir):
    seed_everything(42)

    dm = ClassifDataModule(length=1024)
    model = ClassificationModel()

    _strategy = SingleHPUStrategy()

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, max_steps=10, accelerator=HPUAccelerator(), devices=1, strategy=_strategy
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # TBD enable these tests and remove max_steps
    # # validate
    # result = trainer.validate(datamodule=dm)
    # assert dm.trainer is not None
    # assert result[0]["val_acc"] > 0.7

    # # test
    # result = trainer.test(model, datamodule=dm)
    # assert dm.trainer is not None
    # test_result = result[0]["test_acc"]
    # assert test_result > 0.6

    # # test saved model
    # model_path = os.path.join(tmpdir, "model.pt")
    # trainer.save_checkpoint(model_path)

    # model = ClassificationModel.load_from_checkpoint(model_path)

    # trainer = Trainer(default_root_dir=tmpdir, accelerator=HPUAccelerator(), devices=1, strategy=_strategy)

    # result = trainer.test(model, datamodule=dm)
    # saved_result = result[0]["test_acc"]
    # assert saved_result == test_result


def test_stages_correct(tmpdir):
    """Ensure all stages correctly are traced correctly by asserting the output for each stage."""

    class StageModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            loss = loss.get("loss")
            # tracing requires a loss value that depends on the model.
            # force it to be a value but ensure we use the loss.
            loss = (loss - loss) + torch.tensor(1)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            x = loss.get("x")
            x = (x - x) + torch.tensor(2)
            return {"x": x}

        def test_step(self, batch, batch_idx):
            loss = super().test_step(batch, batch_idx)
            y = loss.get("y")
            y = (y - y) + torch.tensor(3)
            return {"y": y}

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            output = super().predict_step(batch, batch_idx)
            return (output - output) + torch.tensor(4)

    class TestCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["loss"].item() == 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["x"].item() == 2

        def on_test_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["y"].item() == 3

        def on_predict_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert torch.all(outputs == 4).item()

    model = StageModel()
    _strategy = SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=_strategy,
        callbacks=TestCallback(),
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.validate(model)
    trainer.predict(model)


def test_accelerator_is_hpu():
    trainer = Trainer(accelerator=HPUAccelerator(), devices=1, strategy=SingleHPUStrategy())
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == 1


def test_accelerator_with_single_device():
    trainer = Trainer(accelerator="hpu", devices=1)
    assert isinstance(trainer.strategy, SingleHPUStrategy)
    assert isinstance(trainer.accelerator, HPUAccelerator)


@pytest.mark.skipif(device_count() <= 1, reason="Test requires multiple HPU devices")
def test_accelerator_with_multiple_devices(hpus):
    if hpus <= 1:
        pytest.skip(reason="Test reqruires multiple cards")
    trainer = Trainer(accelerator="hpu", devices=hpus)
    assert isinstance(trainer.strategy, HPUParallelStrategy)
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == hpus

    trainer = Trainer(accelerator="hpu")
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == HPUAccelerator.auto_device_count()


@pytest.mark.skipif(device_count() <= 1, reason="Test requires multiple HPU devices")
def test_accelerator_auto_with_devices_hpu(hpus):
    if hpus <= 1:
        pytest.skip(reason="Test reqruires multiple cards")
    trainer = Trainer(accelerator="auto", devices=hpus)
    assert isinstance(trainer.strategy, HPUParallelStrategy)
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == hpus


def test_strategy_choice_single_strategy():
    trainer = Trainer(strategy=SingleHPUStrategy(device=torch.device("hpu")), accelerator=HPUAccelerator(), devices=1)
    assert isinstance(trainer.strategy, SingleHPUStrategy)

    trainer = Trainer(accelerator="hpu", devices=1)
    assert isinstance(trainer.strategy, SingleHPUStrategy)


@pytest.mark.skipif(device_count() <= 1, reason="Test requires multiple HPU devices")
def test_strategy_choice_parallel_strategy(hpus):
    if hpus <= 1:
        pytest.skip(reason="Test reqruires multiple cards")
    trainer = Trainer(
        strategy=HPUParallelStrategy(parallel_devices=[torch.device("hpu")] * hpus),
        accelerator=HPUAccelerator(),
        devices=hpus,
    )
    assert isinstance(trainer.strategy, HPUParallelStrategy)

    trainer = Trainer(accelerator="hpu", devices=hpus)
    assert isinstance(trainer.strategy, HPUParallelStrategy)


def test_devices_auto_choice_hpu():
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.num_devices == HPUAccelerator.auto_device_count()


@pytest.mark.parametrize("hpus", [1])
def test_inference_only(tmpdir, hpus):
    model = BoringModel()

    _strategy = SingleHPUStrategy()
    if hpus > 1:
        parallel_hpus = [torch.device("hpu")] * hpus
        _strategy = HPUParallelStrategy(parallel_devices=parallel_hpus)
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator=HPUAccelerator(), devices=hpus, strategy=_strategy
    )
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


def test_hpu_auto_device_count():
    assert HPUAccelerator.auto_device_count() == HPUAccelerator.auto_device_count()


def test_hpu_unsupported_device_type():
    with pytest.raises(MisconfigurationException, match="`devices` for `HPUAccelerator` must be int, string or None."):
        Trainer(accelerator=HPUAccelerator(), devices=[1])


def test_strategy_params_with_hpu_parallel_strategy():
    bucket_cap_mb = 100
    gradient_as_bucket_view = True
    static_graph = True
    find_unused_parameters = True
    strategy = HPUParallelStrategy(
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        find_unused_parameters=find_unused_parameters,
    )
    assert strategy._ddp_kwargs["bucket_cap_mb"] == bucket_cap_mb
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] == gradient_as_bucket_view
    assert strategy._ddp_kwargs["static_graph"] == static_graph
    assert strategy._ddp_kwargs["find_unused_parameters"] == find_unused_parameters


def test_multi_optimizers_with_hpu(tmpdir):
    class MultiOptimizerModel(BoringModel):
        def configure_optimizers(self):
            opt_a = torch.optim.Adam(self.layer.parameters(), lr=0.001)
            opt_b = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            return opt_a, opt_b

        def training_step(self, batch, batch_idx):
            opt1, opt2 = self.optimizers()
            loss = self.loss(self.step(batch))
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            loss = self.loss(self.step(batch))
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

    model = MultiOptimizerModel()
    model.automatic_optimization = False
    model.val_dataloader = None
    _strategy = SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=_strategy,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)


def test_hpu_device_stats_monitor():
    hpu_stats = HPUAccelerator().get_device_stats("hpu")
    fields = [
        "Limit",
        "InUse",
        "MaxInUse",
        "NumAllocs",
        "NumFrees",
        "ActiveAllocs",
        "MaxAllocSize",
        "TotalSystemAllocs",
        "TotalSystemFrees",
        "TotalActiveAllocs",
    ]
    for f in fields:
        assert any(f in h for h in hpu_stats)


class BaseBM(BoringModel):
    """Model to test with reduce ops."""

    def __init__(self, reduce_op=None):
        """Init."""
        super().__init__()
        self.reduce_op = reduce_op
        self.logged_value_start = 42

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Each ddp process logs 3 values: 42, 43, and 44.
        # logger performs reduce depending on the reduce_op
        loss = super().training_step(batch, batch_idx)
        self.log(
            "logged_value",
            self.logged_value_start,
            prog_bar=True,
            sync_dist=True,
            reduce_fx=self.reduce_op,
            on_epoch=True,
        )
        self.logged_value_start += 1
        return loss


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        """Init."""
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Copy trainer metrics."""
        metric = copy.deepcopy(trainer.logged_metrics)
        self.metrics.append(metric)


class MockHPUParallelStrategy(HPUParallelStrategy):
    def __init__(
        self,
        reduce_op="sum",
        **kwargs: Any,
    ):
        super().__init__()
        self.reduce_op = reduce_op
        self.logged_messages = []

    def reduce(
        self, tensor: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
    ) -> torch.Tensor:
        return super().reduce(tensor, group, self.reduce_op)


@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
def test_hpu_parallel_reduce_op_strategy_default():
    """Test default reduce_op."""
    strategy = MockHPUParallelStrategy()
    # Assert that the strategy's reduce_op attribute is set to the default "sum"
    assert strategy.reduce_op == "sum"


@pytest.mark.standalone()
@pytest.mark.skipif(HPUAccelerator.auto_device_count() < 2, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize(
    ("reduce_op", "expectation"),
    [
        ("sum", nullcontext()),
        ("max", nullcontext()),
        ("min", nullcontext()),
        ("mean", nullcontext()),
        (
            "product",
            pytest.raises(
                TypeError,
                match=f"Unsupported ReduceOp product. Supported ops in HCCL are: {', '.join(supported_reduce_ops)}",
            ),
        ),
        (ReduceOp.SUM, nullcontext()),
        (ReduceOp.MIN, nullcontext()),
        (ReduceOp.MAX, nullcontext()),
        (ReduceOp.AVG, nullcontext()),
        (
            ReduceOp.PRODUCT,
            pytest.raises(
                TypeError,
                match=(
                    "Unsupported ReduceOp RedOpType.PRODUCT. "
                    f"Supported ops in HCCL are: {', '.join(supported_reduce_ops)}"
                ),
            ),
        ),
    ],
    ids=[
        "sum",
        "max",
        "min",
        "mean",
        "product",
        "ReduceOp.SUM",
        "ReduceOp.MIN",
        "ReduceOp.MAX",
        "ReduceOp.AVG",
        "ReduceOp.PRODUCT",
    ],
)
def test_reduce_op_strategy(tmpdir, hpus, reduce_op, expectation):
    """Tests all reduce in HPUParallel strategy."""
    seed_everything(42)
    _model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=MockHPUParallelStrategy(reduce_op=reduce_op),
        max_epochs=1,
        fast_dev_run=3,
        plugins=HPUPrecisionPlugin(precision="bf16-mixed"),
    )
    with expectation:
        trainer.fit(_model)


@pytest.mark.parametrize(
    ("reduce_op", "logged_value_epoch", "logged_value_step"),
    [
        # Epoch = Sum(42, 43, 44), Step = Sum(44)
        ("sum", 129.0, 44.0),
        # Epoch = Max(42, 43, 44), Step = Max(44)
        ("max", 44.0, 44.0),
        # Epoch = Min(42, 43, 44), Step = Min(44)
        ("min", 42.0, 44.0),
        # Epoch = Mean(42, 43, 44), Step = Mean(44)
        ("mean", 43.0, 44.0),
    ],
)
def test_reduce_op_logging(tmpdir, hpus, reduce_op, logged_value_epoch, logged_value_step):
    """Test reduce_op with logger and sync_dist."""
    # Logger has its own reduce_op sanity check.
    # It only accepts following string reduce_ops {min, max, mean, sum}
    # Each ddp process logs 3 values: 42, 43, and 44.
    # logger performs reduce depending on the reduce_op
    if reduce_op == "sum":
        logged_value_epoch *= hpus
        logged_value_step *= hpus

    seed_everything(42)
    _model = BaseBM(reduce_op=reduce_op)
    _strategy = HPUParallelStrategy(parallel_devices=[torch.device("hpu")] * hpus) if hpus > 1 else SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=_strategy,
        max_epochs=1,
        fast_dev_run=3,
        plugins=HPUPrecisionPlugin(precision="bf16-mixed"),
        callbacks=[MetricsCallback()],
    )
    trainer.fit(_model)

    assert torch.allclose(
        trainer.callback_metrics.get("logged_value_epoch"), torch.tensor(logged_value_epoch), atol=1e-4
    )
    assert torch.allclose(trainer.callback_metrics.get("logged_value_step"), torch.tensor(logged_value_step), atol=1e-4)
