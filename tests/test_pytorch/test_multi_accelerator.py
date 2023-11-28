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

import pytest
import torch
from lightning_habana.utils.hpu_distributed import supported_reduce_ops
from lightning_habana.utils.resources import device_count
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
from lightning_habana.pytorch.strategies import HPUParallelStrategy, SingleHPUStrategy


@pytest.fixture()
def _skip_module():
    if "HABANA_VISIBLE_MODULES" in os.environ:
        mod_ids = os.environ["HABANA_VISIBLE_MODULES"]
        if mod_ids == "0,1":
            pytest.skip("Distributed test disabled for modules 0,1")


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


@pytest.mark.skip(reason="TBD : Fix pytest issues")
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
        strategy=MockHPUParallelStrategy(reduce_op=reduce_op, start_method="spawn"),
        max_epochs=1,
        fast_dev_run=3,
        plugins=HPUPrecisionPlugin(precision="bf16-mixed"),
    )
    with expectation:
        trainer.fit(_model)


@pytest.mark.skip(reason="TBD : Fix pytest issues")
@pytest.mark.skipif(HPUAccelerator.auto_device_count() < 2, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize(
    ("reduce_op", "logged_value_epoch", "logged_value_step"),
    [
        # Epoch = Sum(42, 43, 44) * 2, Step = 44 * 2 (for 2 ddp processes)
        ("sum", 258.0, 88.0),
        # Epoch = Max(42, 43, 44), Step = Max(44, ... (x2))
        ("max", 44.0, 44.0),
        # Epoch = Min(42, 43, 44), Step = Min(44, ... (x2))
        ("min", 42.0, 44.0),
        # Epoch = Mean(42(x2), 43(x2), 44(x2)), Step = Mean(44, ... (x2))
        ("mean", 43.0, 44.0),
    ],
)
def test_reduce_op_logging(tmpdir, hpus, reduce_op, logged_value_epoch, logged_value_step):
    """Test reduce_op with logger and sync_dist."""
    # Logger has its own reduce_op sanity check.
    # It only accepts following string reduce_ops {min, max, mean, sum}
    # Each ddp process logs 3 values: 42, 43, and 44.
    # logger performs reduce depending on the reduce_op
    seed_everything(42)
    _model = BaseBM(reduce_op=reduce_op)

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=HPUParallelStrategy(start_method="spawn"),
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


@pytest.mark.usefixtures("_skip_module")
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
