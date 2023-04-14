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
import shutil

import pytest
import torch.distributed
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pytorch_lightning.utilities.imports import _KINETO_AVAILABLE

from lightning_habana.pytorch.accelerator import HPUAccelerator

if _KINETO_AVAILABLE:
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler


@pytest.fixture()
def get_device_count(pytestconfig):
    hpus = int(pytestconfig.getoption("hpus"))
    if not hpus:
        assert HPUAccelerator.auto_device_count() >= 1
        return 1
    assert hpus <= HPUAccelerator.auto_device_count(), "More hpu devices asked than present"
    assert hpus == 1 or hpus % 8 == 0
    return hpus


@pytest.fixture()
def _check_distributed(get_device_count):
    if get_device_count <= 1:
        pytest.skip("Distributed test does not run on single HPU")


def test_hpu_simple_profiler_instances(get_device_count):
    trainer = Trainer(
        profiler="simple",
        accelerator="hpu",
        devices=get_device_count,
    )
    assert isinstance(trainer.profiler, SimpleProfiler)


def test_hpu_simple_profiler_trainer_stages(tmpdir, get_device_count):
    model = BoringModel()
    profiler = SimpleProfiler(dirpath=os.path.join(tmpdir, "profiler_logs"), filename="profiler")
    trainer = Trainer(
        profiler=profiler,
        accelerator="hpu",
        devices=get_device_count,
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)

    actual = set(os.listdir(profiler.dirpath))
    expected = {f"{stage}-profiler.txt" for stage in ("fit", "validate", "test", "predict")}
    assert actual == expected
    for file in list(os.listdir(profiler.dirpath)):
        assert os.path.getsize(os.path.join(profiler.dirpath, file)) > 0


def test_hpu_advanced_profiler_instances(get_device_count):
    trainer = Trainer(
        profiler="advanced",
        accelerator="hpu",
        devices=get_device_count,
    )
    assert isinstance(trainer.profiler, AdvancedProfiler)


def test_hpu_advanced_profiler_trainer_stages(tmpdir, get_device_count):
    model = BoringModel()
    profiler = AdvancedProfiler(dirpath=os.path.join(tmpdir, "profiler_logs"), filename="profiler")
    trainer = Trainer(
        profiler=profiler,
        accelerator="hpu",
        devices=get_device_count,
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)

    actual = set(os.listdir(profiler.dirpath))
    expected = {f"{stage}-profiler.txt" for stage in ("fit", "validate", "test", "predict")}
    assert actual == expected
    for file in list(os.listdir(profiler.dirpath)):
        assert os.path.getsize(os.path.join(profiler.dirpath, file)) > 0


@pytest.mark.usefixtures("_check_distributed")
def test_simple_profiler_distributed_files(tmpdir, get_device_count):
    """Ensure the proper files are saved in distributed."""
    profiler = SimpleProfiler(dirpath="profiler_logs", filename="profiler")
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy="hpu_parallel",
        accelerator="hpu",
        devices=get_device_count,
        profiler=profiler,
        logger=False,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    expected = {
        f"{stage}-profiler-{rank}.txt"
        for stage in ("fit", "validate", "test")
        for rank in range(0, trainer.num_devices)
    }
    # Wait till all hpus have finished.
    # assert fail before a torch.distributed.barrier() keeps the barrier waiting
    # on a process that has returned, causing the test to hang.
    # Do assert checks after the torch.distributed.barrier()
    assert_fail = ""
    torch.distributed.barrier()
    try:
        if trainer.local_rank == 0:
            actual = set(os.listdir(profiler.dirpath))
            assert actual == expected
            for profilerfile in os.listdir(profiler.dirpath):
                with open(os.path.join(profiler.dirpath, profilerfile), encoding="utf-8") as pf:
                    assert len(pf.read()) != 0
    except AssertionError as ae:
        assert_fail = ae
    finally:
        torch.distributed.barrier()
        assert assert_fail == ""
        if trainer.local_rank == 0:
            shutil.rmtree("profiler_logs", ignore_errors=True)


@pytest.mark.usefixtures("_check_distributed")
def test_advanced_profiler_distributed_files(tmpdir, get_device_count):
    """Ensure the proper files are saved in distributed."""
    profiler = AdvancedProfiler(dirpath="profiler_logs", filename="profiler")
    model = BoringModel()
    profiler = AdvancedProfiler(dirpath="profiler_logs", filename="profiler")
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy="hpu_parallel",
        accelerator="hpu",
        devices=get_device_count,
        profiler=profiler,
        logger=False,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    expected = {
        f"{stage}-profiler-{rank}.txt"
        for stage in ("fit", "validate", "test")
        for rank in range(0, trainer.num_devices)
    }
    # Wait till all hpus have finished.
    # assert fail before a torch.distributed.barrier() keeps the barrier waiting
    # on a process that has returned, causing the test to hang.
    # Do assert checks after the torch.distributed.barrier()
    assert_fail = ""
    torch.distributed.barrier()
    try:
        if trainer.local_rank == 0:
            actual = set(os.listdir(profiler.dirpath))
            assert actual == expected
            for profilerfile in os.listdir(profiler.dirpath):
                with open(os.path.join(profiler.dirpath, profilerfile), encoding="utf-8") as pf:
                    assert len(pf.read()) != 0
    except AssertionError as ae:
        assert_fail = ae
    finally:
        torch.distributed.barrier()
        assert assert_fail == ""
        if trainer.local_rank == 0:
            shutil.rmtree("profiler_logs", ignore_errors=True)


def test_hpu_profiler_no_string_instances():
    with pytest.raises(MisconfigurationException) as e_info:
        Trainer(profiler="hpu", accelerator="hpu", devices=1)
    assert "it can only be one of ['simple', 'advanced', 'pytorch', 'xla']" in str(e_info)


def test_hpu_trace_event_cpu_op(tmpdir):
    # Run model and prep json
    model = BoringModel()

    trainer = Trainer(
        accelerator="hpu",
        devices=1,
        max_epochs=1,
        default_root_dir=tmpdir,
        profiler=HPUProfiler(dirpath=tmpdir),
        limit_train_batches=2,
        limit_val_batches=0,
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
                if event["cat"] == "cpu_op":
                    event_duration_arr.append(event["dur"])
            except KeyError:
                pass
        if len(event_duration_arr) == 0:
            raise Exception("Could not find event cpu_op in trace")
        for event_duration in event_duration_arr:
            assert event_duration >= 0


def test_hpu_trace_event_runtime(tmpdir):
    # Run model and prep json
    model = BoringModel()

    trainer = Trainer(
        accelerator="hpu",
        devices=1,
        max_epochs=1,
        default_root_dir=tmpdir,
        profiler=HPUProfiler(dirpath=tmpdir),
        limit_train_batches=2,
        limit_val_batches=0,
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
                if event["cat"] == "Runtime":
                    event_duration_arr.append(event["dur"])
            except KeyError:
                pass
        if len(event_duration_arr) == 0:
            raise Exception("Could not find event hpu_op in trace")
        for event_duration in event_duration_arr:
            assert event_duration >= 0


def test_hpu_trace_event_kernel(tmpdir):
    # Run model and prep json
    model = BoringModel()
    trainer = Trainer(
        accelerator="hpu",
        devices=1,
        max_epochs=1,
        default_root_dir=tmpdir,
        profiler=HPUProfiler(dirpath=tmpdir),
        limit_train_batches=2,
        limit_val_batches=0,
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
                if event["cat"] == "Kernel":
                    event_duration_arr.append(event["dur"])
            except KeyError:
                pass
        if len(event_duration_arr) == 0:
            raise Exception("Could not find event kernel in trace")
        for event_duration in event_duration_arr:
            assert event_duration >= 0
