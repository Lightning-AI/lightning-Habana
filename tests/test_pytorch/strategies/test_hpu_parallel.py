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
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.plugins import CheckpointIO
    from lightning.pytorch.strategies import StrategyRegistry
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.plugins import CheckpointIO
    from pytorch_lightning.strategies import StrategyRegistry

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.pytorch.strategies import HPUParallelStrategy


def test_hpu_parallel_strategy_init():
    """Test init HPUParallelStrategy."""
    strategy = HPUParallelStrategy(parallel_devices=[torch.device("hpu")])

    assert strategy.strategy_name == "hpu_parallel"
    assert strategy.accelerator is None
    assert strategy.parallel_devices == [torch.device("hpu")]
    assert strategy.root_device == torch.device("hpu")
    assert isinstance(strategy.checkpoint_io, HPUCheckpointIO)
    assert strategy._get_process_group_backend() == "hccl"
    assert strategy.cluster_environment is None
    assert strategy._start_method == "spawn"
    assert strategy._timeout == default_pg_timeout
    assert strategy._num_nodes == 1


def test_hpu_parallel_parallel_devices():
    """Test parallel_devices set."""
    devices = [torch.device("hpu")] * 2
    strategy = HPUParallelStrategy(parallel_devices=devices)
    assert len(strategy.parallel_devices) == 2
    assert all(device.type == "hpu" for device in strategy.parallel_devices)


@pytest.mark.standalone_only()
def test_hpu_parallel_broadcast():
    """Broadcasting an object."""
    strategy = HPUParallelStrategy(
        accelerator=HPUAccelerator(),
        parallel_devices=[torch.device("hpu")],
        cluster_environment=LightningEnvironment(),
    )
    strategy.setup_environment()
    obj = torch.randn(2, 2)
    result = strategy.broadcast(obj)

    assert torch.equal(result, obj)


def test_hpu_parallel_custom_strategy_registry():
    """Test custom parallel strategy registry."""

    class CustomCPIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path):
            pass

        def load_checkpoint(self, path):
            pass

        def remove_checkpoint(self, path):
            pass

    class CustomParallelStrategy(HPUParallelStrategy):
        strategy_name = "custom_hpu_parallel"

    StrategyRegistry.register(
        "hpu_parallel_custom_strategy",
        CustomParallelStrategy,
        description="custom HPU Parallel strategy",
        checkpoint_io=CustomCPIO(),
    )
    trainer = Trainer(strategy="hpu_parallel_custom_strategy", accelerator=HPUAccelerator(), devices=1)
    assert isinstance(trainer.strategy, CustomParallelStrategy)
    assert isinstance(trainer.strategy.checkpoint_io, CustomCPIO)
    assert trainer.strategy.strategy_name == "custom_hpu_parallel"


def test_hpu_tensor_init_context():
    """Test that the module under the init-context gets moved to the right device."""
    strategy = HPUParallelStrategy(parallel_devices=[torch.device("hpu")], cluster_environment=LightningEnvironment())
    with strategy.tensor_init_context():
        module = torch.nn.Linear(2, 2)
    assert module.weight.device.type == module.bias.device.type == "hpu"


@pytest.mark.parametrize("strategy_class_name", ["HPUParallelStrategy", "CustomParallelStrategy"])
def test_hpu_parallel_setup_environment(strategy_class_name):
    """Tests setup_distributed is called correctly."""
    strategy = HPUParallelStrategy()
    strategy.__class__.__name__ = strategy_class_name
    with patch.object(strategy, "setup_hccl_env") as mock_setup_hccl_env, patch.object(
        strategy, "setup_distributed"
    ) as mock_setup_distributed, patch(
        "lightning_habana.pytorch.strategies.parallel.ParallelStrategy.setup_environment"
    ):
        strategy.setup_environment()
        mock_setup_hccl_env.assert_called_once()
        if strategy_class_name == "HPUParallelStrategy":
            mock_setup_distributed.assert_called_once()
        else:
            mock_setup_distributed.assert_not_called()


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.standalone_only()
@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_hpu_parallel_strategy_trainer_stages(tmpdir, stage):
    """Test trainer stages with hpu_parallel_strategy."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=2,
        strategy=HPUParallelStrategy(parallel_devices=[torch.device("hpu")] * 2),
        fast_dev_run=True,
    )
    with nullcontext():
        trainer_fn = getattr(trainer, stage)
        trainer_fn(model)


class BaseBM(BoringModel):
    """Model to test with strategy.reduce."""

    def __init__(self, logged_value_arr, reduce_op=None):
        """Init."""
        super().__init__()
        self.logged_value_arr = logged_value_arr
        self.reduce_op = reduce_op

    def training_step(self, batch, batch_idx):
        """Training step."""
        logged_value = self.logged_value_arr[self.trainer.strategy.local_rank]
        reduced_value = self.trainer.strategy.reduce(logged_value, reduce_op=self.reduce_op)
        self.log("reduced_value", reduced_value)
        return super().training_step(batch, batch_idx)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.standalone_only()
@pytest.mark.parametrize(
    "reduce_op",
    [
        "sum",
        "max",
        "min",
        "mean",
    ],
)
def test_hpu_parallel_reduce(tmpdir, reduce_op):
    """Test reduce_op with logger and sync_dist."""
    seed_everything(42)
    logged_value_arr = [torch.rand(1) for _ in range(2)]
    torch_function = getattr(torch, reduce_op)
    expected_value = torch_function(torch.stack(logged_value_arr))

    seed_everything(42)
    _model = BaseBM(logged_value_arr=logged_value_arr, reduce_op=reduce_op)
    _strategy = HPUParallelStrategy(parallel_devices=[torch.device("hpu")] * 2)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=2,
        strategy=_strategy,
        fast_dev_run=True,
    )
    trainer.fit(_model)
    assert expected_value.item() == trainer.callback_metrics.get("reduced_value")
