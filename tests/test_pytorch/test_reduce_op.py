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
# limitations under the License


import torch
import copy
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Trainer, Callback, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.fabric.utilities.types import ReduceOp
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from lightning_fabric.utilities.types import ReduceOp

from contextlib import contextmanager
from typing import Any, Optional, Union
from torch import Tensor
from lightning_habana.pytorch import HPUAccelerator, HPUParallelStrategy, HPUPrecisionPlugin

import pytest


@contextmanager
def does_not_raise():
    """No-op context manager as a complement to pytest.raises."""
    yield


class BaseBM(BoringModel):
    """Model to test with reduce ops."""

    def __init__(self, reduce_op=None):
        """Init."""
        super(BaseBM, self).__init__()
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
        super(MockHPUParallelStrategy, self).__init__()
        self.reduce_op = reduce_op
        self.logged_messages = []
        print(f"{self.reduce_op=}")

    def reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
    ) -> Tensor:
        return super().reduce(tensor, group, self.reduce_op)


@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
def test_hpu_parallel_strategy_default_reduce_op():
    """Test default reduce_op."""
    strategy = MockHPUParallelStrategy()
    # Assert that the strategy's reduce_op attribute is set to the default "sum"
    assert strategy.reduce_op == "sum"


@pytest.mark.skipif(HPUAccelerator.auto_device_count() < 8, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize(
    ("reduce_op", "expectation"),
    [
        ("sum", does_not_raise()),
        ("max", does_not_raise()),
        ("min", does_not_raise()),
        ("mean", does_not_raise()),
        (
            "product",
            pytest.raises(
                AssertionError,
                match="Unsupported ReduceOp product. Only 'sum', 'min', and 'max' are supported with HCCL",
            ),
        ),
        (ReduceOp.SUM, does_not_raise()),
        (ReduceOp.MIN, does_not_raise()),
        (ReduceOp.MAX, does_not_raise()),
        (ReduceOp.AVG, does_not_raise()),
        (
            ReduceOp.PRODUCT,
            pytest.raises(
                AssertionError,
                match="Unsupported ReduceOp RedOpType.PRODUCT. Only 'sum', 'min', and 'max' are supported with HCCL",
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
def test_reduce_op_strategy(tmpdir, reduce_op, expectation):
    """Tests all reduce in HPUParallel strategy."""
    seed_everything(42)
    _model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=8,
        strategy=MockHPUParallelStrategy(reduce_op=reduce_op, start_method="spawn"),
        max_epochs=1,
        fast_dev_run=3,
    )
    with expectation:
        trainer.fit(_model)


@pytest.mark.skipif(HPUAccelerator.auto_device_count() < 8, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize(
    ("reduce_op", "logged_value_epoch", "logged_value_step"),
    [
        # Epoch = Sum(42, 43, 44) * 8, Step = 44 * 8 (for 8 ddp processes)
        ("sum", 1032.0, 352.0),
        # Epoch = Max(42, 43, 44), Step = Max(44, ... (x8))
        ("max", 44.0, 44.0),
        # Epoch = Min(42, 43, 44), Step = Min(44, ... (x8))
        ("min", 42.0, 44.0),
        # Epoch = Mean(42(x8), 43(x8), 44(x8)), Step = Mean(44, ... (x8))
        ("mean", 43.0, 44.0),
    ],
)
def test_reduce_op_logging(tmpdir, reduce_op, logged_value_epoch, logged_value_step):
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
        devices=8,
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
