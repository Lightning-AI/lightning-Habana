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

import os
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from lightning_utilities import module_available
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import always_wrap_policy, wrap
from torch.utils.data import DataLoader

if module_available("lightning"):
    from lightning.fabric import Fabric
    from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
    from lightning.fabric.wrappers import _FabricOptimizer
elif module_available("pytorch_lightning"):
    from lightning_fabric import Fabric
    from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
    from lightning_fabric.wrappers import _FabricOptimizer

from lightning_habana.fabric.accelerator import HPUAccelerator
from lightning_habana.fabric.plugins.fsdp_precision import HPUFSDPPrecision
from lightning_habana.fabric.strategies.fsdp import HPUFSDPStrategy
from lightning_habana.utils.imports import _LIGHTNING_GREATER_EQUAL_2_3_0

from tests.test_fabric.fabric_helpers import RandomDataset

if not _LIGHTNING_GREATER_EQUAL_2_3_0:
    pytestmark = pytest.mark.skip(reason="The tests require lightning version 2.3.0 or above")


def test_hpu_fsdp_strategy_defaults():
    strategy = HPUFSDPStrategy()
    assert strategy.process_group_backend == "hccl"
    assert len(strategy.parallel_devices) == HPUAccelerator.auto_device_count()


class BasicTrainer:
    """Implements a basic training loop for the end-to-end tests."""

    def __init__(self, fabric):
        self.fabric = fabric
        self.model = self.optimizer = self.dataloader = None

    def get_model(self):
        return nn.Linear(32, 2)

    def step(self, model, batch):
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

    def run(self) -> None:
        with self.fabric.init_module():
            model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model, optimizer = self.fabric.setup(model, optimizer)

        dataloader = DataLoader(RandomDataset(32, 64))
        dataloader = self.fabric.setup_dataloaders(dataloader)

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        model.train()

        data_iter = iter(dataloader)
        batch = next(data_iter)
        loss = self.step(model, batch)
        self.fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()


class _Trainer(BasicTrainer):
    def get_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.num_wrapped = 4
        return model

    def step(self, model, batch):
        wrapped_layers = [m for m in model.modules() if isinstance(m, FullyShardedDataParallel)]
        assert len(wrapped_layers) == self.num_wrapped
        assert (self.num_wrapped == 4) == isinstance(model._forward_module, FullyShardedDataParallel)

        precision = self.fabric._precision
        assert isinstance(precision, HPUFSDPPrecision)
        if precision.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif precision.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {precision.precision}")

        for layer in wrapped_layers:
            assert layer.mixed_precision.param_dtype == param_dtype
            assert layer.mixed_precision.reduce_dtype == reduce_dtype
            assert layer.mixed_precision.buffer_dtype == buffer_dtype

        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))


class _TrainerManualWrapping(_Trainer):
    def get_model(self):
        model = super().get_model()
        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        self.num_wrapped = 2
        return model


@pytest.mark.standalone()
def test_fsdp_train(hpus):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    strategy = HPUFSDPStrategy(
        parallel_devices=[torch.device("hpu")] * hpus,
        auto_wrap_policy=always_wrap_policy,
        precision=HPUFSDPPrecision(precision="bf16-mixed"),
    )

    fabric = Fabric(
        accelerator=HPUAccelerator(),
        strategy=strategy,
        devices=hpus,
    )
    fabric.launch()
    trainer = _Trainer(fabric)
    trainer.run()


@pytest.mark.xfail(run=False, reason="Saving checkpoint is not fully supported.")
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_train_save_load(tmp_path, hpus, manual_wrapping):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    trainer_cls = _TrainerManualWrapping if manual_wrapping else _Trainer
    strategy = HPUFSDPStrategy(
        parallel_devices=[torch.device("hpu")] * hpus,
        auto_wrap_policy=always_wrap_policy,
        precision=HPUFSDPPrecision(precision="bf16-mixed"),
    )

    fabric = Fabric(
        accelerator=HPUAccelerator(),
        strategy=strategy,
        devices=hpus,
    )
    fabric.launch()
    trainer = trainer_cls(fabric)
    trainer.run()

    checkpoint_path = fabric.broadcast(str(tmp_path / "fsdp-checkpoint"))

    params_before = deepcopy(list(trainer.model.parameters()))
    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)
    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # re-init all objects and resume
    strategy = HPUFSDPStrategy(
        parallel_devices=[torch.device("hpu")] * hpus,
        auto_wrap_policy=always_wrap_policy,
        precision=HPUFSDPPrecision(precision="bf16-mixed"),
    )

    fabric = Fabric(
        accelerator=HPUAccelerator(),
        strategy=strategy,
        devices=hpus,
    )
    fabric.launch()
    trainer = trainer_cls(fabric)
    trainer.run()

    # check correctness with loaded state
    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 0}
    metadata = fabric.load(checkpoint_path, state)
    for p0, p1 in zip(params_before, trainer.model.parameters()):
        torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

    # check user data in state reloaded
    assert state["steps"] == 1
    assert not metadata

    # attempt to load a key not in the metadata checkpoint
    state = {"model": trainer.model, "coconut": 11}
    with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
        fabric.load(checkpoint_path, state)

    # `strict=False` ignores the missing key
    state = {"model": trainer.model, "coconut": 11}
    fabric.load(checkpoint_path, state, strict=False)
    assert state["coconut"] == 11


def test_setup_with_orig_params_and_multiple_param_groups(hpus):
    """Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device."""
    strategy = HPUFSDPStrategy(
        parallel_devices=[torch.device("hpu")] * hpus,
        auto_wrap_policy=always_wrap_policy,
        precision=HPUFSDPPrecision(precision="bf16-mixed"),
    )
    fabric = Fabric(
        accelerator=HPUAccelerator(),
        strategy=strategy,
        devices=hpus,
    )
    fabric.launch()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Linear(5, 2, bias=False),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 1e-2},
            {"params": model[1].parameters(), "lr": 1e-6},
        ]
    )

    # set up model and optimizer jointly
    wrapped_model, wrapped_optimizer = fabric.setup(model, optimizer)

    assert fabric.strategy._fsdp_kwargs["use_orig_params"]
    assert isinstance(wrapped_optimizer, _FabricOptimizer)
    assert len(wrapped_optimizer.param_groups) == 2
    for i in range(2):
        layer = wrapped_model._forward_module.module[i]
        assert isinstance(layer, FullyShardedDataParallel)
        assert torch.equal(wrapped_optimizer.param_groups[i]["params"][0], layer.weight)

        # A regular parameter as a view into the flattened parameters
        assert isinstance(layer.weight, torch.nn.Parameter)
        assert not isinstance(layer.weight, FlatParameter)


@pytest.mark.standalone()
@pytest.mark.skipif(HPUAccelerator.auto_device_count() <= 1, reason="Test requires multiple HPU devices")
@pytest.mark.parametrize("move_to_device", [True, False])
def test_setup_module_move_to_device(hpus, move_to_device):
    if hpus != 2:
        pytest.skip(reason="Test requires 2 HPU cards")

    strategy = HPUFSDPStrategy(
        parallel_devices=[torch.device("hpu")] * hpus,
        auto_wrap_policy=always_wrap_policy,
        precision=HPUFSDPPrecision(precision="bf16-mixed"),
    )
    fabric = Fabric(
        accelerator=HPUAccelerator(),
        strategy=strategy,
        devices=hpus,
    )
    fabric.launch()

    model = torch.nn.Linear(10, 10, bias=False)  # total params: 10 * 10 = 100
    fabric_model = fabric.setup_module(model, move_to_device=move_to_device)

    assert len(list(fabric_model.parameters())) == 1

    assert next(fabric_model.parameters()).device == torch.device("hpu", 0)
    assert next(fabric_model.parameters()).numel() == 50
    assert isinstance(next(fabric_model.parameters()), nn.Parameter)

    assert fabric.device == torch.device("hpu")


def test_rewrap_warnings(hpus):
    strategy = HPUFSDPStrategy(
        parallel_devices=[torch.device("hpu")] * hpus,
        auto_wrap_policy=always_wrap_policy,
        precision=HPUFSDPPrecision(precision="bf16-mixed"),
    )
    fabric = Fabric(
        accelerator=HPUAccelerator(),
        strategy=strategy,
        devices=hpus,
    )
    fabric.launch()
    device_hpu = torch.device("hpu")
    with fabric.init_module():
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1), device_id=device_hpu)
        )
    with pytest.warns(match="the model is already wrapped"):
        model = fabric.setup(model)
    assert not isinstance(model._forward_module, FullyShardedDataParallel)
    assert isinstance(model._forward_module[2], FullyShardedDataParallel)

    if not _TORCH_GREATER_EQUAL_2_1:
        return

    with fabric.init_module(empty_init=True):
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1), device_id=device_hpu)
        )
    assert model[0].weight.is_meta
    with pytest.warns(match="there are still parameters on the meta device"):
        fabric_model = fabric.setup(model)
    assert next(fabric_model.parameters()).is_meta
