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

import glob
import json
import os
import platform
from contextlib import nullcontext

import pytest
from lightning_habana.utils.resources import device_count
import torch
from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUDDPStrategy, SingleHPUStrategy
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning import Callback
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
    from lightning.pytorch.profilers.pytorch import _KINETO_AVAILABLE, RegisterRecordFunction
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Callback, Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
    from pytorch_lightning.profilers.pytorch import _KINETO_AVAILABLE, RegisterRecordFunction
    from pytorch_lightning.utilities.exceptions import MisconfigurationException


if _KINETO_AVAILABLE:
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler


@pytest.fixture()
def _check_distributed(get_device_count):
    if get_device_count <= 1:
        pytest.skip("Distributed test does not run on single HPU")


@pytest.mark.parametrize(
    ("profiler_str", "profiler_class", "expectation"),
    [
        ("simple", SimpleProfiler, nullcontext()),
        ("advanced", AdvancedProfiler, nullcontext()),
        (
            "hpu",
            HPUProfiler,
            pytest.raises(
                MisconfigurationException,
                match=r"it can only be one of \['simple', 'advanced', 'pytorch', 'xla'\]",
            ),
        ),
    ],
)
def test_hpu_profiler_instances(profiler_str, profiler_class, expectation):
    with expectation:
        trainer = Trainer(
            profiler=profiler_str,
            accelerator=HPUAccelerator(),
            devices=1,
            strategy=SingleHPUStrategy(),
        )
        assert isinstance(trainer.profiler, profiler_class)


@pytest.mark.parametrize(
    ("profiler"),
    [
        (SimpleProfiler),
        (AdvancedProfiler),
    ],
)
def test_hpu_profiler_trainer_stages(tmpdir, profiler):
    model = BoringModel()
    trainer = Trainer(
        profiler=profiler(dirpath=tmpdir, filename="profiler"),
        accelerator=HPUAccelerator(),
        strategy=SingleHPUStrategy(),
        devices=1,
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)

    actual = set(os.listdir(trainer.profiler.dirpath))
    expected = {f"{stage}-profiler.txt" for stage in ("fit", "validate", "test", "predict")}
    assert actual == expected
    for file in list(os.listdir(trainer.profiler.dirpath)):
        assert os.path.getsize(os.path.join(trainer.profiler.dirpath, file)) > 0


@pytest.mark.standalone()
@pytest.mark.usefixtures("_check_distributed")
@pytest.mark.parametrize(("profiler"), [(SimpleProfiler), (AdvancedProfiler)])
@pytest.mark.skipif(device_count() <= 1, reason="Test requires multiple HPU devices")
def test_profiler_trainer_stages_distributed(tmpdir, profiler, get_device_count):
    """Ensure the proper files are saved in distributed."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy=HPUDDPStrategy(),
        accelerator=HPUAccelerator(),
        devices=get_device_count,
        profiler=profiler(dirpath=tmpdir, filename="profiler"),
        fast_dev_run=True,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)

    actual = set(os.listdir(trainer.profiler.dirpath))
    expected = {f"{stage}-profiler-{trainer.local_rank}.txt" for stage in ("fit", "validate", "test", "predict")}
    assert actual == expected
    for profilerfile in os.listdir(trainer.profiler.dirpath):
        with open(os.path.join(trainer.profiler.dirpath, profilerfile), encoding="utf-8") as pf:
            assert len(pf.read()) != 0


@pytest.mark.parametrize(
    "event_name",
    [
        "cpu_op",
        "Runtime",
        "Kernel"
    ],
)
@pytest.mark.skipif(device_count() <= 1, reason="Test requires multiple HPU devices")
@pytest.mark.xfail(strict=False, reason="TBF: Could not find event kernel in trace")
def test_hpu_trace_event(tmpdir, event_name):
    # Run model and prep json
    model = BoringModel()

    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        default_root_dir=tmpdir,
        profiler=HPUProfiler(dirpath=tmpdir),
        fast_dev_run=5,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # get trace path
    trace_path = glob.glob(os.path.join(tmpdir, "fit*training_step*.json"))[0]

    # Check json dumped
    assert os.path.isfile(trace_path)
    with open(trace_path) as file:
        data = json.load(file)
        assert "traceEvents" in data
        event_duration_arr = []
        for event in data["traceEvents"]:
            try:
                if event["cat"] == event_name:
                    event_duration_arr.append(event["dur"])
            except KeyError:
                pass
        if len(event_duration_arr) == 0:
            raise Exception(f"Could not find event {event_name} in trace")
        for event_duration in event_duration_arr:
            assert event_duration >= 0


@pytest.mark.parametrize(("fn", "step_name"), [("test", "test"), ("validate", "validation"), ("predict", "predict")])
@pytest.mark.parametrize("boring_model_cls", [BoringModel])
def test_hpu_profiler_trainer(fn, step_name, boring_model_cls, tmpdir):
    """Ensure that the profiler can be given to the trainer and test step are properly recorded."""
    pytorch_profiler = HPUProfiler(dirpath=tmpdir, filename="profile", schedule=None)
    model = boring_model_cls()
    model.predict_dataloader = model.train_dataloader
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        max_epochs=1,
        limit_test_batches=2,
        profiler=pytorch_profiler,
    )
    getattr(trainer, fn)(model)

    assert sum(e.name.endswith(f"{step_name}_step") for e in pytorch_profiler.function_events)


def test_hpu_profiler_nested(tmpdir):
    """Ensure that the profiler handles nested context."""
    pytorch_profiler = HPUProfiler(dirpath=tmpdir, filename="profiler", schedule=None)

    with pytorch_profiler.profile("a"):
        a = torch.ones(42)
        with pytorch_profiler.profile("b"):
            b = torch.zeros(42)
        with pytorch_profiler.profile("c"):
            _ = a + b

    pytorch_profiler.describe()

    events_name = {e.name for e in pytorch_profiler.function_events}

    names = {"[pl][profile]a", "[pl][profile]b", "[pl][profile]c"}
    ops = {"add", "empty", "fill_", "ones", "zero_", "zeros"}
    ops = {"aten::" + op for op in ops}

    expected = names.union(ops)
    assert events_name == expected, (events_name, torch.__version__, platform.system())


def test_hpu_profiler_multiple_loggers(tmpdir):
    """Tests HPU profiler is able to write its trace with multiple loggers.

    See https://github.com/Lightning-AI/pytorch-lightning/issues/8157.

    """

    def look_for_trace(trace_dir):
        """Determines if a directory contains a PyTorch trace."""
        return any("trace.json" in filename for filename in os.listdir(trace_dir))

    model = BoringModel()
    loggers = [TensorBoardLogger(save_dir=tmpdir), CSVLogger(tmpdir)]
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        profiler=HPUProfiler(),
        logger=loggers,
        limit_train_batches=5,
        max_epochs=1,
    )
    assert len(trainer.loggers) == 2
    trainer.fit(model)
    assert look_for_trace(tmpdir / "lightning_logs" / "version_0")


def test_register_record_function(tmpdir):
    pytorch_profiler = HPUProfiler(
        export_to_chrome=False,
        dirpath=tmpdir,
        filename="profiler",
        schedule=None,
        on_trace_ready=None,
    )

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1))

    model = TestModel()
    input = torch.rand((1, 1))

    with pytorch_profiler.profile("a"), RegisterRecordFunction(model):
        model(input)

    pytorch_profiler.describe()
    event_names = [e.name for e in pytorch_profiler.function_events]
    assert "[pl][module]torch.nn.modules.container.Sequential: layer" in event_names
    assert "[pl][module]torch.nn.modules.linear.Linear: layer.0" in event_names
    assert "[pl][module]torch.nn.modules.activation.ReLU: layer.1" in event_names
    assert "[pl][module]torch.nn.modules.linear.Linear: layer.2" in event_names


def test_hpu_profiler_teardown(tmpdir):
    """This test checks if profiler teardown method is called when trainer is exiting."""

    class TestCallback(Callback):
        def on_fit_end(self, trainer, *args, **kwargs) -> None:
            # describe sets it to None
            assert trainer.profiler._output_file is None

    profiler = HPUProfiler(dirpath=tmpdir, filename="profiler")
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        fast_dev_run=1,
        profiler=profiler,
        callbacks=[TestCallback()],
    )
    trainer.fit(model)

    assert profiler._output_file is None


def test_hpu_profile_callbacks(tmpdir):
    """Checks if profiling callbacks works correctly, specifically when there are two of the same callback type."""
    profiler = HPUProfiler(dirpath=tmpdir, filename="profiler")
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        fast_dev_run=1,
        profiler=profiler,
        callbacks=[EarlyStopping("val_loss"), EarlyStopping("train_loss")],
    )
    trainer.fit(model)
    assert sum(
        e.name == "[pl][profile][Callback]EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}.on_validation_start"
        for e in profiler.function_events
    )
    assert sum(
        e.name == "[pl][profile][Callback]EarlyStopping{'monitor': 'train_loss', 'mode': 'min'}.on_validation_start"
        for e in profiler.function_events
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_hpu_profiler_env(monkeypatch):
    monkeypatch.setenv("HABANA_PROFILE", "1")
    with pytest.raises(AssertionError, match="`HABANA_PROFILE` should not be set when using `HPUProfiler`"):
        HPUProfiler()
