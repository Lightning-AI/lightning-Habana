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
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Callback, LightningModule, Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Callback, LightningModule, Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
from lightning_habana.pytorch.strategies.single import SingleHPUStrategy


@pytest.fixture()
def hmp_params(request):
    return {
        "opt_level": "O1",
        "verbose": False,
        "bf16_file_path": request.config.getoption("--hmp-bf16"),
        "fp32_file_path": request.config.getoption("--hmp-fp32"),
    }


def test_precision_plugin(hmp_params):
    plugin = HPUPrecisionPlugin(precision="bf16-mixed", **hmp_params)
    assert plugin.precision == "bf16-mixed"


def test_mixed_precision(tmpdir, hmp_params: dict):
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
        plugins=[HPUPrecisionPlugin(precision="bf16-mixed", **hmp_params)],
        callbacks=TestCallback(),
    )
    assert isinstance(trainer.strategy, SingleHPUStrategy)
    assert isinstance(trainer.strategy.precision_plugin, HPUPrecisionPlugin)
    assert trainer.strategy.precision_plugin.precision == "bf16-mixed"
    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_unsupported_precision_plugin():
    with pytest.raises(ValueError, match=r"accelerator='hpu', precision='mixed'\)` is not supported."):
        HPUPrecisionPlugin(precision="mixed")