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
import re
from contextlib import nullcontext
from unittest.mock import patch

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


from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
from lightning_habana.pytorch.plugins.precision import _PRECISION_INPUT
from lightning_habana.pytorch.strategies import HPUDDPStrategy, SingleHPUStrategy
from lightning_habana.utils.imports import _HPU_SYNAPSE_GREATER_1_18_0
from lightning_habana.utils.resources import get_device_name_from_hlsmi

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
            assert tengine.fp8.FP8GlobalStateManager.is_fp8_enabled()
            assert not torch.hpu.is_autocast_hpu_enabled()
        else:
            assert not tengine.fp8.FP8GlobalStateManager.is_fp8_enabled()
            assert torch.hpu.is_autocast_hpu_enabled()
        return super().forward(x)


@pytest.mark.parametrize("precision_plugin", [False, True])
def test_autocast_enable_disable(tmpdir, precision_plugin):
    """Tests autocast granular control with HPUPrecisionPlugin."""

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

    precision_plugin = HPUPrecisionPlugin(precision="bf16-mixed") if precision_plugin else None
    assert run_training(tmpdir, BMAutocastGranularControl(), precision_plugin) is not None


@pytest.mark.xfail(strict=False, reason="Env needs to be set")
@pytest.mark.skipif(_HPU_SYNAPSE_GREATER_1_18_0, reason="Test valid for Synapse version <= 1.18.0")
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


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize("replace_layers", [True, False])
def test_hpu_precision_replace_layerse(replace_layers):
    """Tests plugin init with replcae_layers."""
    model = BaseBM()
    plugin = HPUPrecisionPlugin(precision="fp8")
    plugin.convert_modules(model, replace_layers=replace_layers)
    assert isinstance(model.layer, tengine.Linear) == replace_layers


def test_hpu_precision_fp8_not_available_gaudi():
    """Tests fp8 training not supported on Gaudi devices."""
    with patch("lightning_habana.pytorch.plugins.precision.is_fp8_available", return_value=("", False)), pytest.raises(
        NotImplementedError, match="fp8 not supported"
    ):
        HPUPrecisionPlugin(precision="fp8")


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above")
def test_hpu_precision_init_fp8_inference_no_inc():
    with patch("lightning_habana.pytorch.plugins.precision._INTEL_NEURAL_COMPRESSOR_AVAILABLE", False):
        precision = HPUPrecisionPlugin(precision="fp8")
        assert not precision.fp8_inference_available
        assert precision.fp8_training_available


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above")
@pytest.mark.parametrize(
    ("neural_compressor_available", "synapse_version_greater_1_17_0", "expectation"),
    [
        (True, True, nullcontext()),
        (True, False, pytest.raises(OSError, match="FP8 inference on HPU requires SynapsesAI >= 1.17.0")),
        (
            False,
            True,
            pytest.raises(
                ModuleNotFoundError,
                match="Intel neural compressor not found. Install it using `pip install neural-compressor`",
            ),
        ),
    ],
)
def test_hpu_precision_convert_modules_fp8_inference_dependancy(
    tmpdir, neural_compressor_available, synapse_version_greater_1_17_0, expectation
):
    with patch(
        "lightning_habana.pytorch.plugins.precision._INTEL_NEURAL_COMPRESSOR_AVAILABLE", neural_compressor_available
    ), patch(
        "lightning_habana.pytorch.plugins.precision._HPU_SYNAPSE_GREATER_EQUAL_1_17_0", synapse_version_greater_1_17_0
    ):
        precision = HPUPrecisionPlugin(precision="fp8")
        assert precision.fp8_inference_available is (neural_compressor_available and synapse_version_greater_1_17_0)
        with expectation:
            precision.convert_modules(BoringModel(), inference=True, quant=False, fp8_data_path=tmpdir)


def test_hpu_precision_convert_modules_precision_not_fp8():
    precision = HPUPrecisionPlugin(precision="bf16-mixed")
    with pytest.raises(
        AssertionError,
        match=re.escape("HPUPrecisionPlugin.convert_modules() should only be used with precision=`fp8`."),
    ):
        precision.convert_modules(BoringModel(), inference=False)


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize("clean_folder", [os.path.join(os.environ["HABANA_LOGS"], "inc_output")], indirect=True)
@pytest.mark.parametrize("patch_path", ["tmpdir", None])
@pytest.mark.parametrize(
    "fp8_config",
    [
        (str(importlib.resources.path("lightning_habana.pytorch.plugins.quant_config.fp8", "maxabs_measure.json"))),
        (
            {
                "mode": "MEASURE",
                "observer": "maxabs",
                "allowlist": {"types": [], "names": []},
                "blocklist": {"types": [], "names": []},
            }
        ),
    ],
)
def test_hpu_precision_fp8_patch(patch_path, tmpdir, fp8_config, clean_folder):
    """Tests fp8 jsons are patched correctly."""
    model = BaseBM()
    plugin = HPUPrecisionPlugin(precision="fp8")
    patch_path = patch_path if patch_path is None else tmpdir
    plugin.convert_modules(module=model, inference=True, quant=fp8_config, fp8_data_path=patch_path)
    fp8_data_dump_path = os.environ.get("HABANA_LOGS") if patch_path is None else patch_path

    data = fp8_config
    if not isinstance(fp8_config, dict):
        with open(fp8_config, encoding="utf-8") as jfile:
            data = json.load(jfile)

    assert data["dump_stats_path"] == os.path.join(fp8_data_dump_path, "inc_output", "measure")


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize(
    ("params", "expectation"),
    [
        (
            {"inference": True, "quant": True},
            pytest.raises(FileNotFoundError, match=r"Failed to load file"),
        ),
        (
            {"inference": True, "quant": False},
            nullcontext(),
        ),
        (
            {"inference": True, "quant": None},
            pytest.raises(TypeError, match="`quant` must be either a bool, file path or a dictionary."),
        ),
        (
            {"inference": True, "quant": "file_path_does_not_exist"},
            pytest.raises(FileNotFoundError),
        ),
        (
            {
                "inference": True,
                "quant": {
                    "mode": "MEASURE",
                    "allowlist": {"types": [], "names": []},
                },
            },
            nullcontext(),
        ),
        ({"inference": False}, nullcontext()),
        ({"inference": False, "replace_layers": True}, nullcontext()),
        ({"inference": False, "replace_layers": True, "recipe": tengine.recipe.DelayedScaling}, nullcontext()),
    ],
)
def test_hpu_precision_convert_modules(params, expectation, tmpdir):
    """Test HPUPrecisionPlugin.convert_modules."""
    model = BaseBM()
    plugin = HPUPrecisionPlugin(precision="fp8")
    with expectation:
        plugin.convert_modules(module=model, fp8_data_path=tmpdir, **params)


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_inference_with_quant_dict(tmpdir):
    measure_dict = {
        "mode": "MEASURE",
        "observer": "maxabs",
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": []},
    }

    quant_dict = {
        "mode": "QUANTIZE",
        "observer": "maxabs",
        "scale_method": "maxabs_hw",
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": ["lm_head"]},
    }

    seed_everything(42)
    model = BoringModel()
    dm = BoringDataModule()
    precision_plugin = HPUPrecisionPlugin(precision="fp8")

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        fast_dev_run=True,
    )

    # Measurement mode
    precision_plugin.convert_modules(module=model, inference=True, quant=measure_dict, fp8_data_path=tmpdir)
    trainer.test(model, dm)

    # Quant mode
    precision_plugin.convert_modules(module=model, inference=True, quant=quant_dict, fp8_data_path=tmpdir)
    trainer.test(model, dm)


@pytest.mark.standalone_only()
@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_inference_log_files(tmpdir):
    log_file = os.path.join(os.environ["HABANA_LOGS"], "inc_log.txt")
    file_size = 0  # if file does not exist
    if os.path.isfile(log_file):
        file_size = os.path.getsize(log_file)  # file exists. log will be appended.

    precision_plugin = HPUPrecisionPlugin(precision="fp8")
    precision_plugin.convert_modules(module=BoringModel(), inference=True, quant=False, fp8_data_path=tmpdir)

    # check log file is created with size > 0
    assert os.path.isfile(log_file)
    assert os.path.getsize(log_file) > file_size


@pytest.mark.standalone_only()
@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_inference_measurement(tmpdir):
    """Tests inference measruement dumps with fp8_inference."""

    def get_fp8_measurement_files(path):
        """Returns a list of fp8 measurement files."""
        assert path is not None
        assert os.path.isdir(path)
        filenames = []
        for file in os.listdir(path):
            filenames.append(file)
        return filenames

    seed_everything(42)

    model = BoringModel()
    plugin = HPUPrecisionPlugin(precision="fp8")
    plugin.convert_modules(module=model, inference=True, quant=False, fp8_data_path=tmpdir)

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
    filenames = get_fp8_measurement_files(os.path.join(tmpdir, "inc_output"))
    expected_data_files = {
        "measure_hooks_maxabs.json",
        "measure_hooks_maxabs_mod_list.json",
        "measure_hooks_maxabs.npz",
    }
    assert set(filenames) == expected_data_files


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_inference_accuracy(tmpdir):
    """Tests fp8_inference accuracy."""
    test_loss = []

    def get_trainer(plugin):
        return Trainer(
            default_root_dir=tmpdir,
            accelerator=HPUAccelerator(),
            devices=1,
            strategy=SingleHPUStrategy(),
            plugins=plugin,
            fast_dev_run=True,
        )

    for precision in ["bf16", "fp8"]:
        seed_everything(42)
        model = BaseBM()
        plugin = HPUPrecisionPlugin(precision=precision)

        if precision == "fp8":
            # Get measurement from portion of data
            plugin.convert_modules(module=model, inference=True, quant=False, fp8_data_path=tmpdir)
            trainer = get_trainer(plugin)
            trainer.test(model)

            # Set module to quantization mode
            plugin.convert_modules(module=model, inference=True, quant=True, fp8_data_path=tmpdir)

        seed_everything(42)
        dm = BoringDataModule()
        trainer = get_trainer(plugin)
        trainer.test(model, dm)
        test_loss.append(trainer.callback_metrics["test_loss"])

    # Compare bf16 and fp8 inference loss
    assert torch.isclose(test_loss[0], test_loss[1], rtol=0.02, atol=0.01)


@pytest.mark.standalone_only()
@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_with_ddp_strategy(tmpdir, arg_hpus):
    """Negative test for fp8 inference not supported with HPUDDPStrategy."""
    model = BoringModel()
    dm = BoringDataModule()
    plugin = HPUPrecisionPlugin(precision="fp8")
    plugin.convert_modules(module=model, inference=True, quant=False, fp8_data_path=tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=arg_hpus,
        strategy=HPUDDPStrategy(),
        plugins=plugin,
    )

    with pytest.raises(NotImplementedError, match="FP8 inference is not supported with HPUDDPStrategy yet !!!"):
        trainer.test(model, dm)


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_fp8_output(tmpdir):
    """Test HPUPrecisionPlugin with module containing both bf16 and fp8 operations."""

    class FP8InOutDtype(BaseBM):
        def forward(self, x):
            x = self.layer(x)
            assert x.dtype == torch.float32
            return x

    plugin = HPUPrecisionPlugin(precision="fp8")
    model = FP8InOutDtype()
    plugin.convert_modules(model, replace_layers=True)

    run_training(tmpdir, model, plugin)


@pytest.mark.skipif(get_device_name_from_hlsmi() != "GAUDI", reason="Negative test for fp8 on Gaudi")
@pytest.mark.parametrize(
    ("precision", "expectation"),
    [
        (
            "fp8",
            pytest.raises(
                NotImplementedError, match="fp8 not supported: FP8 not supported on Gaudi, Gaudi2 or higher required."
            ),
        ),
        (
            "16-mixed",
            pytest.raises(
                NotImplementedError, match="fp16 not supported: FP16 not supported on Gaudi, Gaudi2 or higher required."
            ),
        ),
    ],
)
def test_hpu_precision_not_supported_on_gaudi(precision, expectation):
    """Test fp8 with unsupported Habana device."""
    with expectation:
        HPUPrecisionPlugin(precision=precision)


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (
            MixedPrecision,
            {"device": "hpu", "precision": "bf16-mixed"},
        ),
        (
            HPUPrecisionPlugin,
            {},
        ),
        (
            HPUPrecisionPlugin,
            {"precision": "bf16-mixed"},
        ),
        (
            HPUPrecisionPlugin,
            {"precision": "bf16"},
        ),
        (
            HPUPrecisionPlugin,
            {"precision": "32-true"},
        ),
        (
            HPUPrecisionPlugin,
            {"precision": "32"},
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"precision": "16-mixed"},
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp16 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"precision": "fp8"},
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
)
def test_precision_plugin_init(plugin, params):
    """Tests precision plugins are instantiated correctly."""
    _plugin = plugin(**params)

    # Common params
    assert _plugin.device == "hpu"
    assert _plugin.precision == params.get("precision", "32-true")

    # HPUPrecision specific params
    if isinstance(_plugin, HPUPrecisionPlugin):
        if _plugin.precision == "fp8":
            assert _plugin.fp8_training_available
            assert _plugin.fp8_inference_available
        else:
            assert not _plugin.fp8_training_available
            assert not _plugin.fp8_inference_available


def test_precision_plugin_invalid_precision_init():
    """Tests precision plugins are instantiated correctly."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`Trainer(accelerator='hpu', precision='f16-mixed')` is not supported. "
            f"`precision` must be one of: {supported_precision}."
        ),
    ):
        HPUPrecisionPlugin(precision="f16-mixed")


@pytest.mark.parametrize(
    ("precision"),
    [
        "32",
        "32-true",
        "bf16",
        "bf16-mixed",
        pytest.param(
            "16-mixed",
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp16 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            "fp8",
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
)
def test_hpu_precision_supported_precision(precision):
    """Tests supported precisions with HPU Precision Plugin."""
    with nullcontext():
        HPUPrecisionPlugin(precision=precision)


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (
            MixedPrecision,
            {"device": "hpu", "precision": "bf16-mixed"},
        ),
        (
            HPUPrecisionPlugin,
            {"precision": "bf16-mixed"},
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"precision": "16-mixed"},
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp16 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            HPUPrecisionPlugin,
            {"precision": "fp8"},
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
)
def test_precision_plugin_fit(tmpdir, plugin, params):
    """Tests precision plugins with trainer.fit."""

    class TestCallback(Callback):
        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            assert trainer.precision == params.get("precision", "32-true")
            raise SystemExit

    seed_everything(42)
    _model = BoringModel()
    _plugin = plugin(**params)
    if isinstance(_plugin, HPUPrecisionPlugin) and params.get("precision") == "fp8":
        _plugin.convert_modules(_model, replace_layers=True)

    with pytest.raises(SystemExit):
        run_training(tmpdir, _model, _plugin, TestCallback())


@pytest.mark.parametrize(
    ("model", "plugin", "params"),
    [
        (BMAutocastCM, None, None),
        (BMAutocastDecorator, None, None),
        (BMPluginActive, MixedPrecision, {"device": "hpu", "precision": "bf16-mixed"}),
        (BMPluginActive, HPUPrecisionPlugin, {"precision": "bf16-mixed"}),
        pytest.param(
            BMPluginActive,
            HPUPrecisionPlugin,
            {"precision": "16-mixed"},
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp16 supported on Gaudi2 and above."
            ),
        ),
        pytest.param(
            BMPluginActive,
            HPUPrecisionPlugin,
            {"precision": "fp8"},
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above."
            ),
        ),
    ],
    ids=[
        "TorchAutocast_CM",
        "TorchAutocast_Decorator",
        "MixedPrecision",
        "HPUPrecisionPlugin_bf16",
        "HPUPrecisionPlugin_fp16",
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
        (BaseBM, HPUPrecisionPlugin, {"precision": "bf16-mixed"}),
    ]
    if get_device_name_from_hlsmi() != "GAUDI":
        model_plugin_list.append(
            (BaseBM, HPUPrecisionPlugin, {"precision": "16-mixed"}),
        )

    loss_list = []
    for item in model_plugin_list:
        seed_everything(42)
        model, plugin, params = item
        model = model()
        _plugin = plugin(**params) if plugin and params else None
        loss_list.append(torch.tensor(run_training(tmpdir, model, _plugin)))

    assert all(torch.allclose(loss_list[0], loss_tensor, rtol=1e-2, atol=1e-2) for loss_tensor in loss_list[1:])


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_plugin_fp8_training_accuracy(tmpdir):
    """Test compare training accuracy between fp32 and fp8 precision."""

    class TestModel(BaseBM):
        """Test model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.layer = tengine.Linear(32, 2)

    precision_list = ["32-true", "fp8"]

    loss_list = []

    for precision in precision_list:
        seed_everything(42)
        model = TestModel()
        _plugin = HPUPrecisionPlugin(precision=precision)
        if precision == "fp8":
            _plugin.convert_modules(model)
        loss_list.append(run_training(tmpdir, model, _plugin))

    assert torch.allclose(torch.tensor(loss_list[0][1]), torch.tensor(loss_list[1][1]), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
@pytest.mark.parametrize("precision", ["32-true", "fp8"])
def test_hpu_precision_active_with_te_module(tmpdir, precision):
    """Tests that fp8 precision is only active when HPUPrecision plugin is init with fp8, even if module from.

    transformer engine is used.

    """

    class TestModel(BoringModel):
        """Test model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.layer = tengine.Linear(32, 2)

        def training_step(self, batch, batch_idx):
            """Training step."""
            # fp8 training is only enabled when precision is fp8,
            # even if module used is from transformer engine.
            if precision == "fp8":
                assert tengine.fp8.FP8GlobalStateManager.is_fp8_enabled()
            else:
                assert not tengine.fp8.FP8GlobalStateManager.is_fp8_enabled()
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            """Configure optimizer."""
            from torch.optim.adamw import AdamW

            return AdamW(self.parameters())

    seed_everything(42)
    model = TestModel()
    _plugin = HPUPrecisionPlugin(precision=precision)
    if precision == "fp8":
        _plugin.convert_modules(model, replace_layers=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        plugins=_plugin,
    )
    trainer.fit(model)


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="Native int64 supported on Gaudi2 and above.")
@pytest.mark.standalone_only()
@pytest.mark.parametrize(
    ("int64_support", "expectation"),
    [
        ("False", pytest.raises(RuntimeError, match="Error when trying to cast Long to Int")),
        ("True", nullcontext()),
    ],
)
def test_hpu_precision_long_type(int64_support, expectation):
    """Tests native support for long tensor on G2."""
    os.environ["PT_ENABLE_INT64_SUPPORT"] = int64_support
    with expectation:
        torch.tensor(torch.iinfo(torch.int64).max, dtype=torch.int64, device=torch.device("hpu"))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        pytest.param(
            torch.float8_e5m2,
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above"
            ),
        ),
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above"
            ),
        ),
        pytest.param(
            torch.float16,
            marks=pytest.mark.skipif(
                get_device_name_from_hlsmi() == "GAUDI", reason="fp16 supported on Gaudi2 and above"
            ),
        ),
        torch.float32,
        torch.bfloat16,
        torch.bool,
    ],
)
def test_hpu_supported_dtypes_tensor_creation(dtype):
    """Tests tensors with supported dtypes can be created on hpu."""
    with nullcontext():
        torch.tensor(42, dtype=dtype, device=torch.device("hpu"))


@pytest.mark.parametrize("intype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float32])
def test_hpu_dtypes_op_output_dtype(intype):
    """Test dtypes type promotion."""
    t1 = torch.tensor([[1, 2], [2, 1]], dtype=intype, device=torch.device("hpu"))
    t2 = torch.tensor([[2, 1], [1, 2]], dtype=intype, device=torch.device("hpu"))

    # Operands are promoted as per torch.promote_types
    t3 = t1.mm(t2)
    t4 = t1.add(t2)
    t5 = t1.div(t2)
    assert t3.dtype == torch.promote_types(t1.dtype, t2.dtype)
    assert t4.dtype == torch.promote_types(t1.dtype, t2.dtype)
    # integer div always promoted to float32.
    assert (
        t5.dtype == torch.promote_types(t1.dtype, t2.dtype)
        if t1.is_floating_point() or t2.is_floating_point()
        else torch.float32
    )

    # torch.autocast only affects torch.float16, torch.bfloat16, torch.float32
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
        # Computes in lower precision if operands in (bf16, fp32) else operand dtype
        t3 = t1.mm(t2)
        # Promoted to highest dtype between operands
        t4 = t1.add(t2)
        # Runs in fp32
        t5 = t1.div(t2)

    assert t3.dtype == intype if intype not in (torch.bfloat16, torch.float32) else torch.bfloat16
    assert t4.dtype == intype
    assert t5.dtype == torch.float32


@pytest.mark.parametrize("intype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_hpu_dtypes_compare_cpu_accuracy(intype, tmpdir):
    """Test dtypes type promotion."""

    class TestModel(BaseBM):
        def forward(self, x):
            # Perform some operations in given dtype
            x = x.to(intype)
            identity = torch.eye(x.shape[1], device=x.device, dtype=intype)
            x = torch.addmm(x, x, identity)

            return super().forward(x.to(torch.float32))

    metrics = []
    for accelerator in [HPUAccelerator(), "cpu"]:
        seed_everything(42)
        trainer = Trainer(
            default_root_dir=tmpdir,
            accelerator=accelerator,
            devices=1,
            strategy=SingleHPUStrategy() if isinstance(accelerator, HPUAccelerator) else "auto",
            fast_dev_run=1,
        )

        trainer.fit(TestModel())
        metrics.append(trainer.logged_metrics)

    # Compare metrics between cpu and hpu
    assert torch.isclose(metrics[0].get("train_loss"), metrics[1].get("train_loss"), atol=1e-5, rtol=1e-5)
    assert torch.isclose(metrics[0].get("val_loss"), metrics[1].get("val_loss"), atol=1e-5, rtol=1e-5)


def test_hpu_precision_plugin_grads_dtype(tmpdir):
    """Tests dtype of gradients on hpu match with those on cpu with HPUPrecisionPlugin."""

    class TestModel(BoringModel):
        """Test model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.linear_hook_handle = self.layer.register_full_backward_hook(self.layer_backward_hook)
            self.grad_dict: dict = {}

        def back_hook(self, layer_name, grad_input, grad_output):
            """Back hook."""
            if layer_name not in self.grad_dict:
                self.grad_dict[layer_name] = {}
                self.grad_dict[layer_name]["grad_input"] = []
                self.grad_dict[layer_name]["grad_output"] = []
            self.grad_dict[layer_name]["grad_input"].append(grad_input)
            self.grad_dict[layer_name]["grad_output"].append(grad_output)

        def layer_backward_hook(self, module, grad_input, grad_output):
            """Layer backward hook."""
            assert isinstance(module, torch.nn.Linear)
            self.back_hook("Linear", grad_input, grad_output)

        def forward(self, x):
            """Forward."""
            x.requires_grad_(True)
            return super().forward(x)

    grad_dict = {}
    for accelerator, strategy, precision_plugin in [
        ("cpu", "auto", MixedPrecision(device="cpu", precision="bf16-mixed")),
        (HPUAccelerator(), SingleHPUStrategy(), HPUPrecisionPlugin(precision="bf16-mixed")),
    ]:
        seed_everything(42)
        model = TestModel()
        dm = BoringDataModule()
        trainer = Trainer(
            default_root_dir=tmpdir,
            accelerator=accelerator,
            devices=1,
            strategy=strategy,
            plugins=precision_plugin,
            fast_dev_run=1,
        )

        trainer.fit(model, dm)
        accelerator_str = "hpu" if isinstance(accelerator, HPUAccelerator) else accelerator
        grad_dict[accelerator_str] = model.grad_dict

    for (kcpu, vcpu), (khpu, vhpu) in zip(grad_dict["cpu"]["Linear"].items(), grad_dict["hpu"]["Linear"].items()):
        # Ensure comparing same grad_type grad_input / grad_output for both devices
        assert kcpu == khpu
        for (grad_cpu,), (grad_hpu,) in zip(vcpu, vhpu):
            # Check grad dtype
            assert grad_cpu.dtype == grad_hpu.dtype


@pytest.mark.skipif(get_device_name_from_hlsmi() == "GAUDI", reason="fp8 supported on Gaudi2 and above.")
def test_hpu_precision_plugin_grads_dtype_fp8(tmpdir):
    """Test dtype of gradients when using fp8 training."""

    class TestModel(BoringModel):
        """Test model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.layer = tengine.Linear(32, 2)
            self.linear_hook_handle = self.layer.register_full_backward_hook(self.layer_backward_hook)
            self.grad_dict: dict = {}

        def back_hook(self, layer_name, grad_input, grad_output):
            """Back hook."""
            if layer_name not in self.grad_dict:
                self.grad_dict[layer_name] = {}
                self.grad_dict[layer_name]["grad_input"] = []
                self.grad_dict[layer_name]["grad_output"] = []
            self.grad_dict[layer_name]["grad_input"].append(grad_input)
            self.grad_dict[layer_name]["grad_output"].append(grad_output)

        def layer_backward_hook(self, module, grad_input, grad_output):
            """Layer backward hook."""
            assert isinstance(module, tengine.Linear)
            self.back_hook("Linear", grad_input, grad_output)

        def forward(self, x):
            """Forward."""
            x.requires_grad_(True)
            return super().forward(x)

    seed_everything(42)
    model = TestModel()
    dm = BoringDataModule()
    plugin = HPUPrecisionPlugin(precision="fp8")
    plugin.convert_modules(model, replace_layers=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        plugins=plugin,
        fast_dev_run=1,
    )

    trainer.fit(model, dm)
    for _, v_grad in model.grad_dict["Linear"].items():
        for (grad_tensor,) in v_grad:
            assert grad_tensor.dtype == torch.float32
