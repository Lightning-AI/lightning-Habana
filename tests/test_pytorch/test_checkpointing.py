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


import os

import pytest
import torch
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Callback, Trainer
    from lightning.pytorch.accelerators.cpu import CPUAccelerator
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Callback, Trainer
    from pytorch_lightning.accelerators.cpu import CPUAccelerator
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.strategies.single_device import SingleDeviceStrategy

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUDDPStrategy, SingleHPUStrategy


@pytest.mark.parametrize(
    "checkpointing",
    [True, False],
)
def test_hpu_checkpointing_trainer_init(tmpdir, checkpointing):
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        enable_checkpointing=checkpointing,
    )
    if checkpointing:
        assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
    else:
        assert trainer.checkpoint_callback is None


@pytest.mark.parametrize(
    ("strategy", "devices"),
    [
        (SingleHPUStrategy, 1),
        pytest.param(HPUDDPStrategy, 2, marks=pytest.mark.standalone_only()),
    ],
)
def test_hpu_checkpoint_save(tmpdir, strategy, devices):
    """Tests checkpoint files are created."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=strategy(),
        devices=devices,
        max_steps=1,
    )
    trainer.fit(model)
    assert model.device.type == "cpu"

    ckpt_file = os.path.join(tmpdir, "lightning_logs", "version_0", "checkpoints", "epoch=0-step=1.ckpt")
    assert os.path.isfile(ckpt_file)
    assert os.path.getsize(ckpt_file) > 0


def test_hpu_checkpointing_disabled(tmpdir):
    """Tests checkpoint files are created."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        max_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model)

    ckpt_file = os.path.join(tmpdir, "lightning_logs", "version_0", "checkpoints", "epoch=0-step=1.ckpt")
    assert not os.path.exists(ckpt_file)


def test_hpu_checkpointing_manual_save(tmpdir):
    """Tests checkpoint files are created."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        max_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model)

    ckpt_file = os.path.join(tmpdir, "lightning_logs", "version_0", "checkpoints", "epoch=0-step=1.ckpt")
    assert not os.path.exists(ckpt_file)  # ckpt file not created due to `enable_checkpoining=False`

    trainer.save_checkpoint(filepath=ckpt_file)  # manual save
    assert os.path.isfile(ckpt_file)
    assert os.path.getsize(ckpt_file) > 0


def test_hpu_modelcheckpoint(tmpdir):
    """Tests checkpoint created by ModelCheckpoint callback."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        max_steps=1,
        callbacks=ModelCheckpoint(dirpath=tmpdir, filename="callback-{epoch}-{step}"),
    )
    trainer.fit(model)

    ckpt_file = os.path.join(tmpdir, "callback-epoch=0-step=1.ckpt")
    assert os.path.isfile(ckpt_file)
    assert os.path.getsize(ckpt_file) > 0


def test_hpu_modelcheckpoint_save_resume(tmpdir):
    """Tests checkpoint created by ModelCheckpoint callback."""

    class TestCheckpointCallback(Callback):
        def on_train_step_end(self, trainer, pl_module, outputs):
            """Check for the checkpoint file after every step."""
            ckpt_file = (os.path.join(tmpdir, f"callback-epoch=0-step={trainer.global_step}.ckpt"),)
            assert os.path.isfile(ckpt_file)
            assert os.path.getsize(ckpt_file) > 0

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        max_steps=2,
        callbacks=[
            ModelCheckpoint(dirpath=tmpdir, filename="callback-{epoch}-{step}", every_n_train_steps=1),
            TestCheckpointCallback(),
        ],
    )
    trainer.fit(model)


@pytest.mark.skip("Test fails in lazy mode.")
def test_hpu_model_weights_after_saving_and_loading_checkpoint(tmpdir):
    """Tests model weights are same after saving and loading checkpoint file."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        max_steps=1,
    )
    trainer.fit(model)

    ckpt_file = os.path.join(tmpdir, "lightning_logs", "version_0", "checkpoints", "epoch=0-step=1.ckpt")
    loaded_model = BoringModel.load_from_checkpoint(ckpt_file)

    for param_original, param_loaded in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param_original, param_loaded), "Model weights do not match after loading!"


@pytest.mark.skip("Test fails in lazy mode.")
@pytest.mark.parametrize(
    ("accelerator", "strategy", "devices"),
    [
        (HPUAccelerator, SingleHPUStrategy, 1),
        (CPUAccelerator, SingleDeviceStrategy, 1),
        pytest.param(
            HPUAccelerator,
            HPUDDPStrategy,
            2,
            marks=[
                pytest.mark.standalone_only(),
                pytest.mark.skip("Test may fail in multi tenent scenario"),
            ],
        ),
    ],
)
def test_hpu_resume_training_from_checkpoint(tmpdir, accelerator, strategy, devices):
    """Tests checkpoint save, load and resume training."""
    model = BoringModel()

    # save checkpoint
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        max_steps=1,
    )
    trainer.fit(model)

    # load checkpoint and resume training
    ckpt_file = os.path.join(tmpdir, "lightning_logs", "version_0", "checkpoints", "epoch=0-step=1.ckpt")
    model = BoringModel.load_from_checkpoint(ckpt_file)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=accelerator(),
        strategy=strategy(),
        devices=devices,
        max_steps=1,
    )
    trainer.fit(model)
