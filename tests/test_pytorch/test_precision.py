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

import importlib
import json
import os
from contextlib import nullcontext

import habana_frameworks.torch.hpex.experimental.transformer_engine as tengine
import pytest
import torch
from lightning_utilities import module_available
from typing_extensions import get_args

if module_available("lightning"):
    from lightning.pytorch import Callback, LightningModule, Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
    from lightning.pytorch.plugins import MixedPrecision
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
    from pytorch_lightning.plugins import MixedPrecision

import re

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
from lightning_habana.pytorch.plugins.precision import _PRECISION_INPUT
from lightning_habana.pytorch.strategies import HPUDDPStrategy, SingleHPUStrategy

supported_precision = get_args(_PRECISION_INPUT)


def run_training(tmpdir, model, plugin, callback=None):
    """Runs a model and returns loss."""
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        plugins=plugin,
        callbacks=callback,
    )
    trainer.fit(model)
    return trainer.callback_metrics["val_loss"], trainer.callback_metrics["train_loss"]


class BaseBM(BoringModel):
    """Model to test with precision Plugin."""

    def forward(self, x):
        """Forward."""
        # Input is in fp32
        identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)

        # torch.mm is computed in bf16.
        x = torch.mm(x, identity)

        # torch.nn.Layer is computed in fp8
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = super().training_step(batch, batch_idx)
        self.log("train_loss", loss.get("loss"), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = super().validation_step(batch, batch_idx)
        self.log("val_loss", loss.get("x"), prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss = super().test_step(batch, batch_idx)
        self.log("test_loss", loss.get("y"), prog_bar=True, sync_dist=True)
        return loss


class BMAutocastCM(BaseBM):
    """Model for torch.autocast context manager."""

    def forward(self, x):
        """Forward."""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            assert torch.hpu.is_autocast_hpu_enabled()
            return super().forward(x)


class BMAutocastDecorator(BaseBM):
    """Model for torch.autocast decorator."""

    @torch.autocast(device_type="hpu", dtype=torch.bfloat16)
    def forward(self, x):
        """Forward."""
        assert torch.hpu.is_autocast_hpu_enabled()
        return super().forward(x)


class BMPluginActive(BaseBM):
    """Model to check active autocast CM when using a precision plugin."""

    def forward(self, x):
        """Forward."""
        if self.trainer.precision == "fp8":
            # Tests fp8 is enabled for supported modules.
            assert tengine.fp8.is_fp8_enabled()
        else:
            assert not tengine.fp8.is_fp8_enabled()
        # Test bf16 enabled.
        assert torch.hpu.is_autocast_hpu_enabled()
        return super().forward(x)


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

    assert run_training(tmpdir, BMAutocastGranularControl(), None) is not None


@pytest.mark.xfail(strict=False, reason="Env needs to be set")
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

    run_training(tmpdir, BMAutocastOverride(), None)


@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_synapse_version(monkeypatch):
    """Test fp8 with unsupported Synapse AI version < 1.14.0."""
    import lightning_habana.utils.imports

    monkeypatch.setattr(lightning_habana.utils.imports, "_HPU_SYNAPSE_GREATER_EQUAL_1_14_0", False)
    with pytest.raises(OSError, match="fp8 training requires `Synapse AI release >= 1.14.0`."):
        HPUPrecisionPlugin(device="hpu", precision="fp8")


@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize("replace_layers", [True, False])
def test_hpu_precision_replace_layerse(replace_layers):
    """Tests plugin init with replcae_layers."""
    model = BaseBM()
    plugin = HPUPrecisionPlugin(device="hpu", precision="fp8", replace_layers=replace_layers)
    plugin.convert_modules(model)
    assert replace_layers == any(
        "habana_frameworks.torch.hpex.experimental.transformer_engine" in m.__module__ for m in model.modules()
    )


@pytest.mark.standalone_only()  # HQT cannot be reloaded in same process
@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize(
    ("inference", "quant", "expectation"),
    [
        (True, True, pytest.raises(FileNotFoundError, match=r"Failed to load file")),
        (True, False, nullcontext()),
        (False, True, nullcontext()),
        (False, False, nullcontext()),
    ],
)
def test_hpu_precision_convert_modules(inference, quant, expectation, tmpdir):
    """Test HPUPrecisionPlugin.convert_modules."""
    model = BaseBM()
    plugin = HPUPrecisionPlugin(device="hpu", precision="fp8")
    with expectation:
        plugin.convert_modules(module=model, inference=inference, quant=quant, fp8_data_path=tmpdir)


@pytest.mark.standalone_only()  # HQT cannot be reloaded in same process
@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize("patch_path", ["tmpdir", None])
def test_hpu_precision_fp8_patch(patch_path, tmpdir):
    """Tests fp8 jsons are patched correctly."""
    model = BaseBM()
    plugin = HPUPrecisionPlugin(device="hpu", precision="fp8")
    patch_path = patch_path if patch_path is None else tmpdir
    plugin.convert_modules(module=model, inference=True, quant=False, fp8_data_path=patch_path)

    package_measure_json = str(
        importlib.resources.path("lightning_habana.pytorch.plugins.quant_config.fp8", "maxabs_measure.json")
    )
    fp8_data_dump_path = os.environ.get("HABANA_LOGS") if patch_path is None else patch_path

    # Check json is patched correctly
    with open(package_measure_json, encoding="utf-8") as jfile:
        data = json.load(jfile)
        stats_path = data["dump_stats_path"]
        xlsx_path = data["dump_stats_xlsx_path"]
        assert stats_path == os.path.join(fp8_data_dump_path, "hqt")
        assert xlsx_path == os.path.join(fp8_data_dump_path, "hqt", "fp8stats.xlsx")


@pytest.mark.standalone_only()  # HQT cannot be reloaded in same process
@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_inference_measurement(tmpdir):
    """Tests inference measruement dumps with fp8_inference."""

    def get_fp8_measurement_files(path):
        """Returns a list of hqt files."""
        assert path is not None
        assert os.path.isdir(path)
        filenames = []
        file_path = []
        for file in os.listdir(path):
            if "hqt" in file:
                file_path.append(os.path.join(path, file))
                filenames.append(file)
        return file_path, filenames

    # cleanup measurement files before test, if any
    file_path, _ = get_fp8_measurement_files(os.environ.get("HABANA_LOGS", None))
    if file_path:
        for file in file_path:
            os.remove(file)

    seed_everything(42)

    model = BaseBM()
    plugin = HPUPrecisionPlugin(device="hpu", precision="fp8")
    plugin.convert_modules(module=model, inference=True, quant=False)

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        plugins=plugin,
        fast_dev_run=True,
    )

    trainer.test(model)

    # check measurement files are dumped
    _, filenames = get_fp8_measurement_files(os.environ.get("HABANA_LOGS", None))
    expected_data_files = {"hqt_hooks_maxabs.json", "hqt_hooks_maxabs_mod_list.json", "hqt_hooks_maxabs.npz"}
    assert set(filenames) == expected_data_files


@pytest.mark.standalone_only()  # HQT cannot be reloaded in same process
@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_inference_quantization(tmpdir):
    """Tests fp8_inference."""
    test_loss = []
    for precision in ["bf16", "fp8"]:
        seed_everything(42)
        model = BaseBM()
        plugin = HPUPrecisionPlugin(device="hpu", precision=precision)
        if precision == "fp8":
            plugin.convert_modules(module=model, inference=True, quant=True)

        trainer = Trainer(
            default_root_dir=tmpdir,
            accelerator=HPUAccelerator(),
            devices=1,
            strategy=SingleHPUStrategy(),
            plugins=plugin,
            fast_dev_run=True,
        )

        trainer.test(model)
        test_loss.append(trainer.callback_metrics["test_loss"])

    # Compare bf16 and fp8 inference loss
    assert torch.isclose(test_loss[0], test_loss[1], rtol=0.01, atol=0.01)


@pytest.mark.standalone_only()
@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_with_ddp_strategy(tmpdir):
    """Negative test for fp8 inference not supported with HPUDDPStrategy."""
    model = BoringModel()
    dm = BoringDataModule()
    plugin = HPUPrecisionPlugin(device="hpu", precision="fp8")
    plugin.convert_modules(module=model, inference=True, quant=False)

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=2,
        strategy=HPUDDPStrategy(),
        plugins=plugin,
    )

    with pytest.raises(NotImplementedError, match="FP8 inference is not supported with HPUDDPStrategy yet !!!"):
        trainer.test(model, dm)


@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_output(tmpdir):
    """Test HPUPrecisionPlugin with module containing both bf16 and fp8 operations."""

    class FP8InOutDtype(BaseBM):
        def forward(self, x):
            # for a module that supports fp8,
            # input is downcasted internally to bf16
            # output is in bf16
            x = self.layer(x)
            assert x.dtype == torch.bfloat16
            return x

    plugin = HPUPrecisionPlugin(device="hpu", precision="fp8")
    model = FP8InOutDtype()
    model = plugin.convert_modules(model)

    run_training(tmpdir, model, plugin)


@pytest.mark.skipif(HPUAccelerator.get_device_name() != "GAUDI", reason="Negative test for fp8 on Gaudi")
def test_hpu_precision_fp8_on_gaudi():
    """Test fp8 with unsupported Habana device."""
    with pytest.raises(
        NotImplementedError, match="fp8 not supported: FP8 not supported on Gaudi, Gaudi2 or higher required."
    ):
        HPUPrecisionPlugin(device="hpu", precision="fp8")


def test_hpu_precision_synapse_version(monkeypatch):
    """Test precision plugin init with unsupported Synapse AI version."""
    import lightning_habana.pytorch.plugins.precision

    monkeypatch.setattr(lightning_habana.pytorch.plugins.precision, "_HPU_SYNAPSE_GREATER_EQUAL_1_11_0", False)
    with pytest.raises(OSError, match="HPU precision plugin requires `Synapse AI release >= 1.11.0`."):
        HPUPrecisionPlugin(device="hpu", precision="bf16-mixed")


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (MixedPrecision, {"device": "hpu", "precision": "bf16-mixed"}),
        (HPUPrecisionPlugin, {"device": "hpu", "precision": "bf16-mixed"}),
        (HPUPrecisionPlugin, {"device": "hpu", "precision": "bf16"}),
        (HPUPrecisionPlugin, {"device": "hpu", "precision": "32-true"}),
        (HPUPrecisionPlugin, {"device": "hpu", "precision": "32"}),
        (
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "bf16-mixed", "replace_layers": "True", "recipe": "DelayedScaling"},
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8", "replace_layers": "False"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8", "replace_layers": "True"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8", "recipe": "DelayedScaling"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8", "replace_layers": "True", "recipe": "DelayedScaling"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
)
def test_precision_plugin_init(plugin, params):
    """Tests precision plugins are instantiated correctly."""
    _plugin = plugin(**params)

    # Common params
    assert _plugin.device == "hpu"
    assert _plugin.precision == params.get("precision")

    # HPUPrecision specific params
    if isinstance(_plugin, HPUPrecisionPlugin):
        if _plugin.precision == "fp8":
            assert _plugin.fp8_train_available
            assert _plugin.replace_layers == params.get("replace_layers", False)
            assert _plugin.recipe == params.get("recipe", None)
        else:
            assert not _plugin.fp8_train_available
            assert not _plugin.replace_layers
            assert _plugin.recipe is None


@pytest.mark.parametrize(
    ("precision", "expectation"),
    [
        ("32", nullcontext()),
        ("32-true", nullcontext()),
        ("bf16", nullcontext()),
        ("bf16-mixed", nullcontext()),
        pytest.param(
            "fp8",
            nullcontext(),
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        (
            "fp16",
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"`Trainer(accelerator='hpu', precision='fp16')` is not supported. "
                    f"`precision` must be one of: {supported_precision}."
                ),
            ),
        ),
    ],
)
def test_hpu_precision_supported_precision(precision, expectation):
    """Tests supported precisions with HPU Precision Plugin."""
    with expectation:
        HPUPrecisionPlugin(device="hpu", precision=precision)


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (MixedPrecision, {"device": "hpu", "precision": "bf16-mixed"}),
        (HPUPrecisionPlugin, {"device": "hpu", "precision": "bf16-mixed"}),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8", "replace_layers": "False"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8", "replace_layers": "True", "recipe": "DelayedScaling"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
)
def test_precision_plugin_fit(tmpdir, plugin, params):
    """Tests precision plugins with trainer.fit."""

    class TestCallback(Callback):
        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            assert trainer.precision == params.get("precision")
            raise SystemExit

    seed_everything(42)
    _model = BoringModel()
    _plugin = plugin(**params)
    if isinstance(_plugin, HPUPrecisionPlugin) and params.get("precision") == "fp8":
        _plugin.convert_modules(_model)

    with pytest.raises(SystemExit):
        run_training(tmpdir, _model, _plugin, TestCallback())


@pytest.mark.parametrize(
    ("model", "plugin", "params"),
    [
        (BMAutocastCM, None, None),
        (BMAutocastDecorator, None, None),
        (BMPluginActive, MixedPrecision, {"device": "hpu", "precision": "bf16-mixed"}),
        (BMPluginActive, HPUPrecisionPlugin, {"device": "hpu", "precision": "bf16-mixed"}),
        pytest.param(
            BMPluginActive,
            HPUPrecisionPlugin,
            {"device": "hpu", "precision": "fp8"},
            marks=pytest.mark.skipif(
                HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
    ids=[
        "TorchAutocast_CM",
        "TorchAutocast_Decorator",
        "MixedPrecision",
        "HPUPrecisionPlugin_bf16",
        "HPUPrecisionPlugin_fp8",
    ],
)
def test_mixed_precision_autocast_to_precision_active(tmpdir, model, plugin, params):
    """Tests autocast is active with torch.autocast context manager."""
    seed_everything(42)
    _model = model()
    _plugin = plugin(**params) if plugin and params else None
    if isinstance(_plugin, HPUPrecisionPlugin) and params.get("precision") == "fp8":
        _plugin.convert_modules(_model)
    run_training(tmpdir, _model, _plugin)


def test_mixed_precision_compare_accuracy(tmpdir):
    """Test and compare accuracy for mixed precision training methods."""
    model_plugin_list = [
        (BMAutocastCM, None, None),
        (BMAutocastDecorator, None, None),
        (BaseBM, MixedPrecision, {"device": "hpu", "precision": "bf16-mixed"}),
        (BaseBM, HPUPrecisionPlugin, {"device": "hpu", "precision": "bf16-mixed"}),
    ]
    is_gaudi = HPUAccelerator().get_device_name() == "GAUDI"
    if not is_gaudi:
        model_plugin_list.append(
            (
                BaseBM,
                HPUPrecisionPlugin,
                {"device": "hpu", "precision": "fp8", "replace_layers": "True", "recipe": "DelayedScaling"},
            )
        )

    loss_list = []
    for item in model_plugin_list:
        seed_everything(42)
        model, plugin, params = item
        _plugin = plugin(**params) if plugin and params else None
        BoringDataModule()
        if isinstance(_plugin, HPUPrecisionPlugin) and params.get("precision") == "fp8":
            model = _plugin.convert_modules(model())
        else:
            model = model()
        loss_list.append(run_training(tmpdir, model, _plugin))

    # Assert loss is same for all instances except fp8
    assert all(x == loss_list[0] for x in loss_list[:-1]), list(zip(model_plugin_list, loss_list))
    if not is_gaudi:
        # Assert loss is close between baseline and fp8
        assert torch.allclose(torch.tensor(loss_list[0]), torch.tensor(loss_list[-1]), rtol=0.1, atol=0.1)


@pytest.mark.skipif(HPUAccelerator.get_device_name() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize("precision", ["bf16-mixed", "fp8"])
def test_hpu_precision_active_with_te_module(tmpdir, precision):
    """Tests that fp8 precision is only active when HPUPrecision plugin is init with fp8, even if module from.

    transformer engine is used.

    """

    class TestModel(BoringModel):
        """Test model."""

        def __init__(self):
            """init."""
            super().__init__()
            self.layer = tengine.Linear(32, 2)

        def training_step(self, batch, batch_idx):
            """Training step."""
            # torch.autocast is enabled for both bf16 and fp8
            assert torch.hpu.is_autocast_hpu_enabled()
            # fp8 training is only enabled when precision is fp8,
            # even if module used is from transformer engine.
            if precision == "fp8":
                assert tengine.fp8.is_fp8_enabled()
            else:
                assert not tengine.fp8.is_fp8_enabled()
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            """Configure optimizer."""
            from torch.optim.adamw import AdamW

            return AdamW(self.parameters())

    seed_everything(42)
    model = TestModel()
    _plugin = HPUPrecisionPlugin(device="hpu", precision=precision)
    # HPUPrecisionPlugin.convert_modules not reqiored as self.layer is already a transformer engine module
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        plugins=_plugin,
    )
    trainer.fit(model)
