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


import pytest
import torch
import torch.nn as nn
from lightning.fabric import Fabric, seed_everything

from lightning_habana.fabric.accelerator.accelerator import HPUAccelerator
from lightning_habana.fabric.plugins.precision.precision import HPUPrecision
from lightning_habana.fabric.strategies.single import SingleHPUStrategy
from tests.test_fabric.fabric_helpers import BoringFabric


class MixedPrecisionModule(nn.Module):
    def __init__(self, expected_dtype):
        super().__init__()
        self.expected_dtype = expected_dtype
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        assert x.dtype == self.expected_dtype
        output = self.layer(x)
        assert output.dtype == self.expected_dtype
        return output


class MixedPrecisionBoringFabric(BoringFabric):
    expected_dtype: torch.dtype

    def get_model(self):
        return MixedPrecisionModule(self.expected_dtype)

    def step(self, model, batch):
        assert model.layer.weight.dtype == torch.float32

        assert batch.dtype == torch.float32
        output = model(batch)
        assert output.dtype == torch.float32
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

    def after_backward(self, model, optimizer):
        assert model.layer.weight.grad.dtype == torch.float32


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        pytest.param("bf16-mixed", torch.float32),
        pytest.param("32", torch.float32),
        pytest.param("bf16-mixed", torch.float32),
    ],
)
def test_hpu(precision, expected_dtype):
    fabric = MixedPrecisionBoringFabric(
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        plugins=[HPUPrecision(precision=precision)],
    )
    fabric.expected_dtype = expected_dtype
    fabric.run()


def test_hpu_fused_optimizer():
    def run():
        seed_everything(1234)
        fabric = Fabric(
            accelerator=HPUAccelerator(),
            devices=1,
            strategy=SingleHPUStrategy(),
            plugins=[HPUPrecision(precision="bf16")],
        )

        model = nn.Linear(10, 10).to(fabric.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        model, optimizer = fabric.setup(model, optimizer)

        data = torch.randn(10, 10, device="hpu")
        target = torch.randn(10, 10, device="hpu")

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(data)
            loss = (output - target).abs().sum()
            fabric.backward(loss)
            optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses), model.parameters()

    def run_fused():
        seed_everything(1234)
        fabric = Fabric(
            accelerator=HPUAccelerator(),
            devices=1,
            strategy=SingleHPUStrategy(),
            plugins=[HPUPrecision(precision="bf16")],
        )

        model = nn.Linear(10, 10).to(fabric.device)
        from habana_frameworks.torch.hpex.optimizers import FusedSGD

        optimizer = FusedSGD(model.parameters(), lr=1.0)

        model, optimizer = fabric.setup(model, optimizer)

        data = torch.randn(10, 10, device="hpu")
        target = torch.randn(10, 10, device="hpu")

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(data)
            loss = (output - target).abs().sum()
            fabric.backward(loss)
            optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses), model.parameters()

    losses, params = run()
    losses_fused, params_fused = run_fused()

    torch.testing.assert_close(losses, losses_fused)
    for p, q in zip(params, params_fused):
        torch.testing.assert_close(p, q)
