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
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Callback, LightningModule, Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.plugins import MixedPrecisionPlugin
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.plugins import MixedPrecisionPlugin

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
from lightning_habana.pytorch.strategies.single import SingleHPUStrategy


@pytest.fixture()
def precision_plugin_params():
    """Returns params for PrecisionPlugin."""
    return {"device": "hpu", "precision": "bf16-mixed"}


def run_training(tmpdir, model, plugin, callback=[]):
    """Runs a model and returns loss."""
    _model = model()
    _strategy = SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=_strategy,
        plugins=plugin,
        callbacks=callback,
    )
    trainer.fit(_model)
    return trainer.callback_metrics["val_loss"].to(torch.bfloat16), trainer.callback_metrics["train_loss"].to(
        torch.bfloat16
    )


class BaseBM(BoringModel):
    """Model to test with precision Plugin."""

    def forward(self, x):
        """Forward."""
        # Downcasting is lazy.
        # Operands will be downcasted if operator supports bfloat16
        assert x.dtype == torch.float32
        identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        x = torch.mm(x, identity)
        assert x.dtype == torch.bfloat16
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = super().training_step(batch, batch_idx)
        self.log("train_loss", loss.get("loss").to(torch.bfloat16), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = super().validation_step(batch, batch_idx)
        self.log("val_loss", loss.get("x").to(torch.bfloat16), prog_bar=True, sync_dist=True)
        return loss


class BMAutocastCM(BaseBM):
    """Model for torch.autocast context manager."""

    def forward(self, x):
        """Forward."""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            return super().forward(x)


class BMAutocastDecorator(BaseBM):
    """Model for torch.autocast decorator."""

    @torch.autocast(device_type="hpu", dtype=torch.bfloat16)
    def forward(self, x):
        """Forward."""
        return super().forward(x)


@pytest.mark.parametrize(
    ("plugin"),
    [
        (HPUPrecisionPlugin),
        (MixedPrecisionPlugin),
    ],
)
def test_precision_plugin_instance(plugin, precision_plugin_params):
    """Tests precision plugins are instantiated correctly."""
    _plugin = plugin(**(precision_plugin_params))
    assert _plugin.precision == "bf16-mixed"
    assert _plugin.device == "hpu"


@pytest.mark.parametrize(
    ("plugin"),
    [
        (HPUPrecisionPlugin),
        (MixedPrecisionPlugin),
    ],
)
def test_precision_plugin_fit(tmpdir, plugin, precision_plugin_params):
    """Tests precision plugins with trainer.fit."""

    class TestCallback(Callback):
        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            assert trainer.precision == "bf16-mixed"
            raise SystemExit

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),  # TBD- set default in accelerator
        plugins=[plugin(**(precision_plugin_params))],
        callbacks=TestCallback(),
    )
    assert isinstance(trainer.strategy, SingleHPUStrategy)
    assert isinstance(trainer.strategy.precision_plugin, plugin)
    assert trainer.strategy.precision_plugin.precision == "bf16-mixed"
    assert trainer.strategy.precision_plugin.device == "hpu"
    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.parametrize(
    ("model", "plugin", "params"),
    [
        (BMAutocastCM, [], ""),
        (BMAutocastDecorator, [], ""),
        (BaseBM, MixedPrecisionPlugin, "precision_plugin_params"),
        (BaseBM, HPUPrecisionPlugin, "precision_plugin_params"),
    ],
    ids=[
        "TorchAutocast_CM",
        "TorchAutocast_Decorator",
        "MixedPrecisionPlugin",
        "HPUPrecisionPlugin",
    ],
)
def test_mixed_precision_autocast_active(tmpdir, model, plugin, params, request):
    """Tests autocast is active with torch.autocast context manager."""

    class TrainTestCallback(Callback):
        def on_batch_start(self, trainer, pl_module):
            assert torch.hpu.is_autocast_hpu_enabled()

    _model = model
    _plugin = plugin(**request.getfixturevalue(params)) if plugin and params else []
    seed_everything(42)
    run_training(tmpdir, _model, _plugin, [TrainTestCallback()])


@pytest.mark.parametrize(
    "model_plugin_list",
    [
        [
            (BMAutocastCM, [], ""),
            (BMAutocastDecorator, [], ""),
            (BaseBM, MixedPrecisionPlugin, "precision_plugin_params"),
            (BaseBM, HPUPrecisionPlugin, "precision_plugin_params"),
        ],
    ],
    ids=[
        "AutocastCM_AutocastDecorator_MixedPrecisionPlugin_HPUPrecisionPlugin",
    ],
)
def test_mixed_precision_compare_accuracy(tmpdir, model_plugin_list, request):
    """Test and compare accuracy for mixed precision training methods."""
    loss_list = []
    for model, plugin, params in model_plugin_list:
        _plugin = plugin(**request.getfixturevalue(params)) if plugin and params else []
        # Reset seed before each trainer.fit call
        seed_everything(42)
        loss_list.append(run_training(tmpdir, model, _plugin))
    assert all(x == loss_list[0] for x in loss_list), list(zip(model_plugin_list, loss_list))


def test_autocast_enable_disable(tmpdir):
    """Tests autocast control with enabled arg."""

    class BMAutocastGranularControl(BaseBM):
        """Tests autocast control with enabled arg."""

        def forward(self, x):
            """Forward."""
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                # Downcasting is lazy.
                # Operands will be downcasted if operator supports bfloat16
                assert torch.hpu.is_autocast_hpu_enabled()
                assert x.dtype == torch.float32
                identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
                x = torch.mm(x, identity)
                assert x.dtype == torch.bfloat16

                # In the disabled subregion, inputs from the surrounding region
                # should be cast to required dtype before use
                x = x.to(torch.float32)
                identity = identity.to(torch.float32)
                with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=False):
                    assert not torch.hpu.is_autocast_hpu_enabled()
                    x = torch.mm(x, identity)
                    assert x.dtype == torch.float32

                # Re-entering autocast enabled region
                assert torch.hpu.is_autocast_hpu_enabled()
                x = torch.mm(x, identity)
                assert x.dtype == torch.bfloat16
            return self.layer(x)

    assert run_training(tmpdir, BMAutocastGranularControl, []) is not None


@pytest.mark.xfail(reason="Env needs to be set")
def test_autocast_operators_override(tmpdir):
    """Tests operator dtype overriding with torch autocast."""
    # The override lists are set in cmdline

    class BMAutocastOverride(BaseBM):
        """Model to test with precision Plugin."""

        def forward(self, x):
            """Forward."""
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
                x = x.to(torch.float32)
                identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
                # Due to operator override,
                # torch.mm will now operate in torch.float32
                y = torch.mm(x, identity)
                assert y.dtype == torch.float32

                # and torch.tan will operate in torch.bfloat16
                z = torch.tan(x)
                assert z.dtype == torch.bfloat16
            return self.layer(x)

    run_training(tmpdir, BMAutocastOverride, [])
