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
from unittest import mock

import pytest
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
elif module_available("pytorch_lightning"):
    from pytorch_lightning.cli import LightningCLI
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel


@pytest.mark.parametrize(
    ("param", "value", "is_class", "init_kwargs"),
    [
        ("accelerator", "auto", False, None),
        ("accelerator", "hpu", False, None),
        pytest.param(
            "accelerator",
            "lightning_habana.pytorch.accelerator.HPUAccelerator",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("strategy", "auto", False, None),
        ("strategy", "hpu_single", False, None),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.SingleHPUStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("strategy", "hpu_parallel", False, None),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.HPUParallelStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "strategy",
            "hpu_ddp",
            None,
            None,
            marks=pytest.mark.xfail(reason="String strategy `hpu_ddp` not registered with Lightning"),
        ),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.HPUDDPStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "strategy",
            "hpu_deepspeed",
            False,
            None,
            marks=pytest.mark.xfail(reason="String strategy `hpu_deepspeed` not registered with Lightning"),
        ),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.HPUDeepSpeedStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("plugins", "lightning_habana.HPUPrecisionPlugin", True, {"precision": "bf16-mixed"}),
        ("plugins", "lightning_habana.HPUCheckpointIO", True, None),
        pytest.param(
            "profiler",
            "hpu",
            False,
            None,
            marks=pytest.mark.skip(reason="String init of HPUProfiler is not supported"),
        ),
        pytest.param(
            "profiler",
            "lightning_habana.HPUProfiler",
            True,
            None,
            marks=pytest.mark.skip(reason="lightning_habana should be imported before lightning to use HPUProfiler"),
        ),
    ],
)
def test_cli_from_cmdline(param, value, is_class, init_kwargs, arg_hpus):
    """Test LightningCLI with HPU from cmdline."""
    cli_args = [
        "fit",
        "--trainer.fast_dev_run=True",
        f"--trainer.{param}={value}",
        f"--trainer.devices={arg_hpus}",
    ]

    if is_class is True and init_kwargs is not None:
        for key, val in init_kwargs.items():
            cli_args.append(f"--trainer.{param}.{key}={val}")

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, BoringDataModule)

    cli_config = cli.config["fit"].as_dict()

    if is_class is True:
        assert cli_config.get("trainer").get(param).get("class_path") == value
        if init_kwargs is not None:
            assert all(
                cli_config.get("trainer").get(param).get("init_args").get(key) == value
                for key, value in init_kwargs.items()
            )
    else:
        assert cli_config.get("trainer").get(param) == value


@pytest.mark.parametrize(
    ("env", "param", "value", "is_class"),
    [
        ("PL_FIT__TRAINER__ACCELERATOR", "accelerator", "auto", False),
        ("PL_FIT__TRAINER__ACCELERATOR", "accelerator", "hpu", False),
        pytest.param(
            "PL_FIT__TRAINER__ACCELERATOR",
            "accelerator",
            "lightning_habana.pytorch.accelerator.HPUAccelerator",
            True,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("PL_FIT__TRAINER__STRATEGY", "strategy", "auto", False),
        ("PL_FIT__TRAINER__STRATEGY", "strategy", "hpu_single", False),
        pytest.param(
            "PL_FIT__TRAINER__STRATEGY",
            "strategy",
            "lightning_habana.pytorch.strategies.SingleHPUStrategy",
            True,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("PL_FIT__TRAINER__STRATEGY", "strategy", "hpu_parallel", False),
        pytest.param(
            "PL_FIT__TRAINER__STRATEGY",
            "strategy",
            "lightning_habana.pytorch.strategies.HPUParallelStrategy",
            True,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "PL_FIT__TRAINER__STRATEGY",
            "strategy",
            "hpu_ddp",
            None,
            marks=pytest.mark.xfail(reason="String strategy `hpu_ddp` not registered with Lightning"),
        ),
        pytest.param(
            "PL_FIT__TRAINER__STRATEGY",
            "strategy",
            "lightning_habana.pytorch.strategies.HPUDDPStrategy",
            True,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "PL_FIT__TRAINER__STRATEGY",
            "strategy",
            "hpu_deepspeed",
            False,
            marks=pytest.mark.xfail(reason="String strategy `hpu_deepspeed` not registered with Lightning"),
        ),
        pytest.param(
            "PL_FIT__TRAINER__STRATEGY",
            "strategy",
            "lightning_habana.pytorch.strategies.HPUDeepSpeedStrategy",
            True,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "PL_FIT__TRAINER__PLUGINS",
            "plugins",
            "lightning_habana.HPUPrecisionPlugin",
            True,
            marks=pytest.mark.skip(reason="Init args for class init cannot be passed through env"),
        ),  # TODO: check
        ("PL_FIT__TRAINER__PLUGINS", "plugins", "lightning_habana.HPUCheckpointIO", True),
        pytest.param(
            "PL_FIT__TRAINER__PROFILER",
            "profiler",
            "hpu",
            False,
            marks=pytest.mark.skip(reason="String init of HPUProfiler is not supported"),
        ),
        pytest.param(
            "PL_FIT__TRAINER__PROFILER",
            "profiler",
            "lightning_habana.HPUProfiler",
            True,
            marks=pytest.mark.skip(reason="lightning_habana should be imported before ligthning to use HPUProfiler"),
        ),
    ],
)
def test_cli_from_env(monkeypatch, env, param, value, is_class, arg_hpus):
    """Test LightningCLI with HPU with env."""
    cli_args = ["fit", "--trainer.fast_dev_run=True", f"--trainer.devices={arg_hpus}"]

    monkeypatch.setenv(env, value)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, BoringDataModule, parser_kwargs={"default_env": True})

    cli_config = cli.config["fit"].as_dict()

    if is_class is True:
        assert cli_config.get("trainer").get(param).get("class_path") == value
    else:
        assert cli_config.get("trainer").get(param) == value


@pytest.mark.parametrize(
    ("param", "value", "is_class", "init_kwargs"),
    [
        ("accelerator", "auto", False, None),
        ("accelerator", "hpu", False, None),
        pytest.param(
            "accelerator",
            "lightning_habana.pytorch.accelerator.HPUAccelerator",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("strategy", "auto", False, None),
        ("strategy", "hpu_single", False, None),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.SingleHPUStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("strategy", "hpu_parallel", False, None),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.HPUParallelStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "strategy",
            "hpu_ddp",
            None,
            None,
            marks=pytest.mark.xfail(reason="String strategy `hpu_ddp` not registered with Lightning"),
        ),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.HPUDDPStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        pytest.param(
            "strategy",
            "hpu_deepspeed",
            False,
            None,
            marks=pytest.mark.xfail(reason="String strategy `hpu_deepspeed` not registered with Lightning"),
        ),
        pytest.param(
            "strategy",
            "lightning_habana.pytorch.strategies.HPUDeepSpeedStrategy",
            True,
            None,
            marks=pytest.mark.xfail(reason="https://github.com/Lightning-AI/pytorch-lightning/issues/19682"),
        ),
        ("plugins", "lightning_habana.HPUPrecisionPlugin", True, {"precision": "bf16-mixed"}),
        ("plugins", "lightning_habana.HPUCheckpointIO", True, None),
        pytest.param(
            "profiler",
            "hpu",
            False,
            None,
            marks=pytest.mark.skip(reason="String init of HPUProfiler is not supported"),
        ),
        pytest.param(
            "profiler",
            "lightning_habana.HPUProfiler",
            True,
            None,
            marks=pytest.mark.skip(reason="lightning_habana should be imported before ligthning to use HPUProfiler"),
        ),
    ],
)
def test_cli_save_and_load_config(tmpdir, param, value, is_class, init_kwargs, arg_hpus):
    """Save and load a LightningCLI config on HPU."""
    # Config is saved automatically in logdir
    cli_args_config_save = [
        "fit",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=0",
        "--trainer.max_epochs=1",
        f"--trainer.devices={arg_hpus}",
        f"--trainer.{param}={value}",
        "--trainer.log_every_n_steps=1",
        f"--trainer.default_root_dir={tmpdir}",
    ]

    if is_class is True and init_kwargs is not None:
        for key, val in init_kwargs.items():
            cli_args_config_save.append(f"--trainer.{param}.{key}={val}")

    with mock.patch("sys.argv", ["any.py"] + cli_args_config_save):
        cli = LightningCLI(BoringModel, BoringDataModule)

    config_file = os.path.join(tmpdir, "lightning_logs", "version_0", "config.yaml")
    assert os.path.isfile(config_file)

    cli_args_config_load = [
        "fit",
        "--trainer.fast_dev_run=True",
        f"--config={config_file}",
    ]

    with mock.patch("sys.argv", ["any.py"] + cli_args_config_load):
        cli = LightningCLI(BoringModel, BoringDataModule)

    cli_config = cli.config["fit"].as_dict()

    if is_class is True:
        assert cli_config.get("trainer").get(param).get("class_path") == value
        if init_kwargs is not None:
            assert all(
                cli_config.get("trainer").get(param).get("init_args").get(key) == value
                for key, value in init_kwargs.items()
            )
    else:
        assert cli_config.get("trainer").get(param) == value
