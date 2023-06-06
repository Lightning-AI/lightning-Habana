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

from contextlib import contextmanager

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
from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
from lightning_habana.pytorch.strategies import HPUParallelStrategy, SingleHPUStrategy


@contextmanager
def does_not_raise():
    """No-op context manager as a complement to pytest.raises."""
    yield


@contextmanager
def does_not_raise():
    """No-op context manager as a complement to pytest.raises"""
    yield


@pytest.fixture()
def hmp_params(request):
    """Returns params for HPUPrecisionPlugin"""
    return {
        "opt_level": "O1",
        "verbose": False,
        "bf16_file_path": request.config.getoption("--hmp-bf16"),
        "fp32_file_path": request.config.getoption("--hmp-fp32"),
    }


@pytest.fixture()
def mpp_params():
    """Returns params for MixedPrecisionPlugin"""
    return {"device": "hpu"}


def run_training(tmpdir, model, plugin):
    """Runs a model and returns loss"""
    _model = model()
    _strategy = SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=_strategy,
        plugins=plugin,
    )
    trainer.fit(_model)
    return trainer.callback_metrics['val_loss'].to(torch.bfloat16), trainer.callback_metrics['train_loss'].to(torch.bfloat16)


class BaseBM(BoringModel):
    """Model to test with precision Plugin"""

    def forward(self, x):
        """Forward"""
        # Downcasting is lazy.
        # Operands will be downcasted if operator supports bfloat16
        assert x.dtype == torch.float32
        identity = torch.eye(
            x.shape[1], device=x.device, dtype=x.dtype)
        x = torch.mm(x, identity)
        assert x.dtype == torch.bfloat16
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss = super().training_step(batch, batch_idx)
        self.log('train_loss', loss.get('loss').to(torch.bfloat16),
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss = super().validation_step(batch, batch_idx)
        self.log('val_loss', loss.get("x").to(
            torch.bfloat16), prog_bar=True, sync_dist=True)
        return loss


class BaseBMActive(BaseBM):
    """Model with torch.autocast enabled check"""

    def forward(self, x):
        """Forward"""
        assert torch.hpu.is_autocast_hpu_enabled()
        return super().forward(x)


class BMAutocastCM(BaseBMActive):
    """Model for torch.autocast context manager"""

    def forward(self, x):
        """Forward"""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            return super().forward(x)


class BMAutocastDecorator(BaseBMActive):
    """Model for torch.autocast decorator"""
    @torch.autocast(device_type="hpu", dtype=torch.bfloat16)
    def forward(self, x):
        """Forward"""
        return super().forward(x)


@pytest.mark.parametrize('plugin,params', [
    (HPUPrecisionPlugin, "hmp_params"),
    (MixedPrecisionPlugin, "mpp_params"),
])
def test_precision_plugins_instance(plugin, params, request):
    """Tests precision plugins are instantiated correctly"""
    _plugin = plugin(precision="bf16-mixed", **request.getfixturevalue(params))
    assert _plugin.precision == "bf16-mixed"


@pytest.mark.parametrize('plugin,params', [
    (HPUPrecisionPlugin, "hmp_params"),
    (MixedPrecisionPlugin, "mpp_params"),
])
def test_mixed_precision_plugin(tmpdir, plugin, params, request):
    """Tests precision plugins with trainer.fit"""
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
        strategy=SingleHPUStrategy(),  # TBD- set default in accelertor
        plugins=[plugin(precision="bf16-mixed", **
                        request.getfixturevalue(params))],
        callbacks=TestCallback(),
    )
    assert isinstance(trainer.strategy, SingleHPUStrategy)
    assert isinstance(trainer.strategy.precision_plugin, plugin)
    assert trainer.strategy.precision_plugin.precision == "bf16-mixed"
    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_unsupported_precision_plugin():
    """Tests unsupported HPUPrecisionPlugin init"""
    with pytest.raises(ValueError, match=r"accelerator='hpu', precision='mixed'\)` is not supported."):
        HPUPrecisionPlugin(precision="mixed")


@pytest.mark.parametrize(
    "model,plugin,params,expectation",
    [
        (BMAutocastCM, [], "", does_not_raise()),
        (BMAutocastDecorator, [], "", does_not_raise()),
        (BaseBMActive, MixedPrecisionPlugin,
         "mpp_params", does_not_raise()),
        (BaseBMActive, HPUPrecisionPlugin, "hmp_params", pytest.raises(
            AssertionError, match="False = <function is_autocast_hpu_enabled")),
    ],
    ids=[
        "TorchAutocast_CM_True",
        "TorchAutocast_Decorator_True",
        "MixedPrecisionPlugin_True",
        "HPUPrecisionPlugin_False",
    ]
)
def test_mixed_precision_autocast_active(tmpdir, model, plugin, params, expectation, request):
    """Tests autocast is active with torch.autocast context manager"""
    _model = model
    _plugin = plugin
    if plugin and params:
        _plugin = plugin(precision="bf16-mixed", **
                         request.getfixturevalue(params))
    seed_everything(42)
    with expectation:
        assert run_training(tmpdir, _model, _plugin) is not None


@pytest.mark.parametrize(
    "model_plugin_list",
    [
        [(BaseBM, HPUPrecisionPlugin, "hmp_params"),
         (BMAutocastCM, [], ""),],
        [(BaseBM, HPUPrecisionPlugin, "hmp_params"),
         (BMAutocastDecorator, [], ""),],
        [(BaseBM, HPUPrecisionPlugin, "hmp_params"),
         (BaseBM, MixedPrecisionPlugin, "mpp_params"),],
        [(BaseBM, HPUPrecisionPlugin, "hmp_params"),
         (BMAutocastCM, [], ""),
         (BMAutocastDecorator, [], ""),
         (BaseBM, MixedPrecisionPlugin, "mpp_params")],
    ],
    ids=[
        "HPUPrecisionPlugin_AutocastCM",
        "HPUPrecisionPlugin_AutocastDecorator",
        "HPUPrecisionPlugin_MixedPrecisionPlugin",
        "HPUPrecisionPlugin_AutocastCM_AutocastDecorator_MixedPrecisionPlugin",
    ])
def test_mixed_precision_compare_accuracy(tmpdir, model_plugin_list, request):
    loss_list = []
    for model, plugin, params in model_plugin_list:
        _plugin = plugin
        if plugin and params:
            _plugin = plugin(precision="bf16-mixed", **
                             request.getfixturevalue(params))
        # Reset seed before each trainer.fit call
        seed_everything(42)
        loss_list.append(run_training(tmpdir, model, _plugin))
    assert all(x == loss_list[0] for x in loss_list), [
        (item1, item2) for item1, item2 in zip(model_plugin_list, loss_list)]


def test_autocast_enable_disable(tmpdir):
    """Tests autocast control with enabled arg"""
    class BMAutocastGranularControl(BaseBM):
        """Tests autocast control with enabled arg"""

        def forward(self, x):
            """Forward"""
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                # Downcasting is lazy.
                # Operands will be downcasted if operator supports bfloat16
                assert torch.hpu.is_autocast_hpu_enabled()
                assert x.dtype == torch.float32
                identity = torch.eye(
                    x.shape[1], device=x.device, dtype=x.dtype)
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
    """Tests operator dtype overriding with torch autocast"""
    # The override lists are set in cmdline

    class BMAutocastOverride(BaseBM):
        """Model to test with precision Plugin"""

        def forward(self, x):
            """Forward"""
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
                x = x.to(torch.float32)
                identity = torch.eye(
                    x.shape[1], device=x.device, dtype=x.dtype)
                # Due to operator override,
                # torch.mm will now operate in torch.float32
                y = torch.mm(x, identity)
                assert y.dtype == torch.float32

                # and torch.tan will operate in torch.bfloat16
                z = torch.tan(x)
                assert z.dtype == torch.bfloat16
            return self.layer(x)

    run_training(tmpdir, BMAutocastOverride, [])
