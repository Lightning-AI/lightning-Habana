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
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.plugins import CheckpointIO
    from lightning.pytorch.strategies import StrategyRegistry
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.plugins import CheckpointIO
    from pytorch_lightning.strategies import StrategyRegistry

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.pytorch.strategies import HPUDDPStrategy, HPUParallelStrategy


def test_hpu_ddp_strategy_init():
    bucket_cap_mb = 100
    gradient_as_bucket_view = True
    static_graph = True
    find_unused_parameters = True
    strategy = HPUDDPStrategy(
        parallel_devices=[torch.device("hpu")] * 2,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        find_unused_parameters=find_unused_parameters,
    )
    # HPU specific params
    assert strategy._get_process_group_backend() == "hccl"
    assert strategy.root_device == torch.device("hpu")
    assert len(strategy.parallel_devices) == 2
    assert isinstance(strategy.checkpoint_io, HPUCheckpointIO)

    # DDP params
    assert strategy._ddp_kwargs["bucket_cap_mb"] == bucket_cap_mb
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] == gradient_as_bucket_view
    assert strategy._ddp_kwargs["static_graph"] == static_graph
    assert strategy._ddp_kwargs["find_unused_parameters"] == find_unused_parameters


def test_hpu_ddp_strategy_device_not_hpu(tmpdir):
    """Tests hpu required with HPUDDPStrategy."""
    trainer = Trainer(
        default_root_dir=tmpdir, accelerator="cpu", strategy=HPUDDPStrategy(), devices=1, fast_dev_run=True
    )
    with pytest.raises(AssertionError, match="HPUDDPStrategy requires HPUAccelerator"):
        trainer.fit(BoringModel())


def test_hpu_ddp_custom_strategy_registry():
    """Test custom parallel strategy registry."""

    class CustomCPIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path):
            pass

        def load_checkpoint(self, path):
            pass

        def remove_checkpoint(self, path):
            pass

    class CustomDDPStrategy(HPUDDPStrategy):
        strategy_name = "custom_hpu_ddp"

    StrategyRegistry.register(
        "hpu_ddp_custom_strategy",
        CustomDDPStrategy,
        description="custom HPU Parallel strategy",
        checkpoint_io=CustomCPIO(),
    )
    trainer = Trainer(strategy="hpu_ddp_custom_strategy", accelerator=HPUAccelerator(), devices=1)
    assert isinstance(trainer.strategy, CustomDDPStrategy)
    assert isinstance(trainer.strategy.checkpoint_io, CustomCPIO)
    assert trainer.strategy.strategy_name == "custom_hpu_ddp"


def test_hpu_ddp_tensor_init_context():
    """Test that the module under the init-context gets moved to the right device."""
    strategy = HPUDDPStrategy(parallel_devices=[torch.device("hpu")], cluster_environment=LightningEnvironment())
    with strategy.tensor_init_context():
        module = torch.nn.Linear(2, 2)
    assert module.weight.device.type == module.bias.device.type == "hpu"


@pytest.mark.standalone()
@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_hpu_ddp_strategy_trainer_stages(tmpdir, stage, arg_hpus):
    """Test trainer stages with hpu_parallel_strategy."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=arg_hpus,
        strategy=HPUDDPStrategy(parallel_devices=[torch.device("hpu")] * arg_hpus),
        fast_dev_run=True,
    )
    with nullcontext():
        trainer_fn = getattr(trainer, stage)
        trainer_fn(model)


@pytest.mark.standalone()
@pytest.mark.parametrize(
    "reduce_op",
    [
        "sum",
        "max",
        "min",
        "mean",
    ],
)
def test_hpu_ddp_reduce(tmpdir, arg_hpus, reduce_op):
    """Test reduce_op with logger and sync_dist."""
    seed_everything(42)
    logged_value_arr = [torch.rand(1) for _ in range(arg_hpus)]
    torch_function = getattr(torch, reduce_op)
    expected_value = torch_function(torch.stack(logged_value_arr))

    class BaseBM(BoringModel):
        """Model to test with reduce ops."""

        def __init__(self, reduce_op=None):
            """Init."""
            super().__init__()
            self.reduce_op = reduce_op
            self.reduced_value = None
            self.logged_value = None

        def training_step(self, batch, batch_idx):
            """Training step."""
            self.logged_value = logged_value_arr[self.trainer.strategy.local_rank]
            self.reduced_value = self.trainer.strategy.reduce(self.logged_value, reduce_op=reduce_op)
            return super().training_step(batch, batch_idx)

    seed_everything(42)
    _model = BaseBM(reduce_op=reduce_op)
    _strategy = HPUDDPStrategy(parallel_devices=[torch.device("hpu")] * arg_hpus)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=arg_hpus,
        strategy=_strategy,
        fast_dev_run=True,
    )
    trainer.fit(_model)
    assert expected_value.item() == _model.reduced_value.item()


def test_hpu_ddp_setup_distributed():
    """Tests setup_distributed is called from HPUDDPStrategy exactly once and not from HPUParallelStrategy."""
    with patch.object(HPUParallelStrategy, "setup_distributed") as parallel_setup_distributed, patch.object(
        HPUDDPStrategy, "setup_distributed"
    ) as ddp_setup_distributed:
        strategy = HPUDDPStrategy(
            accelerator=HPUAccelerator(),
            parallel_devices=[torch.device("hpu")],
            cluster_environment=LightningEnvironment(),
        )
        strategy.setup_distributed()

        parallel_setup_distributed.assert_not_called()
        ddp_setup_distributed.assert_called_once()
