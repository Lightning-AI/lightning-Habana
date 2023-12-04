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

from enum import Enum

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as f
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer, seed_everything
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule

from lightning_habana.pytorch import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy


class HPUGraphMode(Enum):
    """HPU graph modes."""

    TRAIN_NONE = 0
    # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#training-loop-with-capture-and-replay
    TRAIN_CAPTURE_AND_REPLAY = 1
    # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#training-loop-with-make-graphed-callables
    TRAIN_MAKE_GRAPHED_CALLABLES = 2
    # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#training-loop-with-modulecacher
    TRAIN_MODULECACHER = 3
    INFERENCE_NONE = 4
    # https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html
    INFERENCE_CAPTURE_AND_REPLAY = 5
    # https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html
    INFERENCE_WRAP_IN_HPU_GRAPH = 6


class NetHPUGraphs(LightningModule):
    """Model with modifications to support hpu graphs."""

    def __init__(self, graph_mode=HPUGraphMode.TRAIN_NONE, batch_size=None):
        """Init."""
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.loss = []
        self.graph_mode = graph_mode
        if (
            self.graph_mode == HPUGraphMode.TRAIN_CAPTURE_AND_REPLAY
            or self.graph_mode == HPUGraphMode.INFERENCE_CAPTURE_AND_REPLAY
        ):
            self.g = htcore.hpu.HPUGraph()
            self.g_val = htcore.hpu.HPUGraph()
            self.automatic_optimization = False
            self.training_step = self.train_with_capture_and_replay
            self.static_input = torch.zeros((batch_size), 1, 28, 28, device="hpu")
            self.static_target = torch.zeros((batch_size,), device="hpu", dtype=torch.long)
            self.static_y_pred = torch.zeros((batch_size,), device="hpu", dtype=torch.long)
            self.static_loss = None
            self.acc = None
            self.validation_step = self.validation_step_capture_replay
        else:
            self.training_step = self.training_step_automatic
            self.validation_step = self.validation_step_automatic

    def forward(self, x):
        """Forward."""
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.mm(x, torch.eye(x.shape[1], device=x.device, dtype=x.dtype))
        return f.log_softmax(x, dim=1)

    def training_step_automatic(self, batch, batch_idx):
        """Automatic optimization training step."""
        x, y = batch
        loss = f.cross_entropy(self.forward(x), y)
        self.log("train_loss", loss)
        return loss

    def train_with_capture_and_replay(self, batch, batch_idx):
        """Training step with hpu graphs capture and replay."""
        optimizer = self.optimizers()
        if batch_idx == 0 and self.current_epoch == 0:
            # First we warmup
            optimizer.zero_grad(set_to_none=True)
            y_pred = self.forward(self.static_input)
            loss = f.cross_entropy(y_pred, self.static_target)
            loss.backward()
            optimizer.step()
            return loss
        if batch_idx == 1 and self.current_epoch == 0:
            # Then we capture
            optimizer.zero_grad(set_to_none=True)
            with htcore.hpu.graph(self.g):
                static_y_pred = self(self.static_input)
                self.static_loss = f.cross_entropy(static_y_pred, self.static_target)
                self.static_loss.backward()
                optimizer.step()
                return self.static_loss
        # Finally replay the captured graph
        # data must be copied to existing tensors that were used in the capture phase
        data, target = batch
        self.static_input.copy_(data)
        self.static_target.copy_(target)
        self.g.replay()
        self.log("train_loss", self.static_loss)
        # result is available in static_loss tensor after graph is replayed
        return self.static_loss

    def validation_step_capture_replay(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        if batch_idx == 0 and self.current_epoch == 0:
            probs = self(self.static_input)
            self.acc = self.accuracy(probs, self.static_target)
        if batch_idx == 1 and self.current_epoch == 0:
            with htcore.hpu.graph(self.g_val):
                probs = self(self.static_input)
                self.acc = self.accuracy(probs, self.static_target)
        else:
            self.static_input.copy_(x)
            self.static_target.copy_(y)
            self.g_val.replay()
            self.log("val_acc", self.acc)

    def validation_step_automatic(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        if self.graph_mode == HPUGraphMode.INFERENCE_CAPTURE_AND_REPLAY:
            if batch_idx == 0:
                with htcore.hpu.graph(self.g):
                    static_y_pred = self.forward(self.static_input)
                    self.static_loss = f.cross_entropy(static_y_pred, self.static_target)
            else:
                self.static_input.copy_(x)
                self.static_target.copy_(y)
                self.g.replay()
                self.log("test_acc", self.accuracy(None, y, self.static_y_pred))
        else:
            self.log("test_acc", self.accuracy(self.forward(x), y))

    @staticmethod
    def accuracy(logits, y, pred=None):
        """Calculate accuracy."""
        if pred is not None:
            return torch.sum(torch.eq(pred, y).to(torch.float32)) / len(y)
        return torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=0.1)


def train_model(root_dir, hpus, model, data_module, profiler=None, mode="fit"):
    """Runs trainer.<fit / validate>."""
    seed_everything(42)
    _strategy = SingleHPUStrategy()
    if hpus > 1:
        _strategy = HPUParallelStrategy()
    trainer = Trainer(
        default_root_dir=root_dir,
        accelerator=HPUAccelerator(),
        profiler=profiler,
        devices=hpus,
        strategy=_strategy,
        max_epochs=1,
        fast_dev_run=3,
    )
    if hasattr(trainer, mode):
        func = getattr(trainer, mode)
    func(model, data_module)
    return trainer.logged_metrics


def get_model(graph_mode):
    """Returns model instances depending on HPU graph mode."""
    if graph_mode == HPUGraphMode.TRAIN_NONE:
        return NetHPUGraphs(graph_mode=graph_mode)
    if graph_mode == HPUGraphMode.TRAIN_CAPTURE_AND_REPLAY:
        return NetHPUGraphs(graph_mode=graph_mode, batch_size=200)
    if graph_mode == HPUGraphMode.TRAIN_MAKE_GRAPHED_CALLABLES:
        model = NetHPUGraphs(graph_mode=graph_mode).to(torch.device("hpu"))
        x = torch.randn(200, 1, 28, 28, device="hpu")
        return htcore.hpu.make_graphed_callables(model, (x,))
    if graph_mode == HPUGraphMode.TRAIN_MODULECACHER:
        model = NetHPUGraphs(graph_mode=graph_mode)
        return htcore.hpu.ModuleCacher(max_graphs=10)(model=model, inplace=True)
    if graph_mode == HPUGraphMode.INFERENCE_NONE:
        return NetHPUGraphs(graph_mode=graph_mode)
    if graph_mode == HPUGraphMode.INFERENCE_CAPTURE_AND_REPLAY:
        return NetHPUGraphs(graph_mode=graph_mode, batch_size=200)
    if graph_mode == HPUGraphMode.INFERENCE_WRAP_IN_HPU_GRAPH:
        model = NetHPUGraphs(graph_mode=graph_mode).to(torch.device("hpu"))
        return htcore.hpu.wrap_in_hpu_graph(model, asynchronous=False, disable_tensor_cache=True)
    return None


@pytest.mark.parametrize(
    ("graph_mode", "mode"),
    [
        (HPUGraphMode.TRAIN_NONE, "fit"),
        (HPUGraphMode.TRAIN_CAPTURE_AND_REPLAY, "fit"),
        (HPUGraphMode.TRAIN_MAKE_GRAPHED_CALLABLES, "fit"),
        (HPUGraphMode.TRAIN_MODULECACHER, "fit"),
        (HPUGraphMode.INFERENCE_NONE, "validate"),
        (HPUGraphMode.INFERENCE_CAPTURE_AND_REPLAY, "validate"),
        (HPUGraphMode.INFERENCE_WRAP_IN_HPU_GRAPH, "validate"),
    ],
)
def test_hpu_graphs(tmpdir, graph_mode, mode):
    """Trains with a given HPU graph mode."""
    seed_everything(42)
    model = get_model(graph_mode)
    data_module = MNISTDataModule(batch_size=200)
    data_module.val_dataloader = data_module.train_dataloader
    train_model(tmpdir, 1, model=model, data_module=data_module, profiler=None, mode=mode)


@pytest.mark.parametrize(
    "train_modes",
    [
        [(HPUGraphMode.TRAIN_NONE), (HPUGraphMode.TRAIN_CAPTURE_AND_REPLAY)],
        [(HPUGraphMode.TRAIN_NONE), (HPUGraphMode.TRAIN_MAKE_GRAPHED_CALLABLES)],
        [(HPUGraphMode.TRAIN_NONE), (HPUGraphMode.TRAIN_MODULECACHER)],
    ],
    ids=[
        "baseline_vs_capture_and_replay",
        "baseline_vs_make_graphed_callables",
        "baseline_vs_modulecacher",
    ],
)
def test_hpu_graph_accuracy_train(tmpdir, train_modes):
    loss_metrics = []
    for graph_mode in train_modes:
        seed_everything(42)
        hpu_graph_model = get_model(graph_mode)
        data_module = MNISTDataModule(batch_size=200)
        loss_metrics.append(train_model(tmpdir, 1, model=hpu_graph_model, data_module=data_module, profiler=None))
    assert torch.allclose(
        loss_metrics[0]["train_loss"], loss_metrics[1]["train_loss"], rtol=0.05
    ), loss_metrics  # Compare train loss
    assert torch.allclose(
        loss_metrics[0]["val_acc"], loss_metrics[1]["val_acc"], rtol=0.05
    ), loss_metrics  # Compare val acc


@pytest.mark.parametrize(
    "train_modes",
    [
        [(HPUGraphMode.INFERENCE_NONE), (HPUGraphMode.INFERENCE_CAPTURE_AND_REPLAY)],
        [(HPUGraphMode.INFERENCE_NONE), (HPUGraphMode.INFERENCE_WRAP_IN_HPU_GRAPH)],
    ],
    ids=[
        "baseline_vs_capture_and_replay",
        "baseline_vs_wrap_in_hpu_graphs",
    ],
)
def test_hpu_graph_accuracy_inference(tmpdir, train_modes):
    loss_metrics = []
    for graph_mode in train_modes:
        seed_everything(42)
        hpu_graph_model = get_model(graph_mode)
        data_module = MNISTDataModule(batch_size=200)
        loss_metrics.append(
            train_model(tmpdir, 1, model=hpu_graph_model, data_module=data_module, mode="test", profiler=None)
        )
    assert torch.allclose(
        loss_metrics[0]["test_acc"], loss_metrics[1]["test_acc"], rtol=0.05
    ), loss_metrics  # Compare test acc


def test_automatic_optimization_graph_capture(tmpdir):
    """Test to showcase HPU Graphs with automatic optimization."""

    class ManualCaptureModel(NetHPUGraphs):
        """Test model to capture HPU Graphs manually."""

        def __init__(self, graph_mode=HPUGraphMode.TRAIN_CAPTURE_AND_REPLAY):
            super().__init__()
            self.automatic_optimization = True

        def on_train_batch_start(self, batch, batch_idx):
            """Encapsulate complete training step Start capture here."""
            if batch_idx == 0 and self.current_epoch == 0:
                # Warmup
                pass
            elif batch_idx == 0 and self.current_epoch == 0:
                # Manual Capture begin
                self.g.capture_begin()

        def on_train_batch_end(self, outputs, batch, batch_idx):
            """Encapsulate complete training step Stop capture here."""
            if batch_idx == 0 and self.current_epoch == 0:
                # Warmup end
                pass
            elif batch_idx == 0 and self.current_epoch == 0:
                # Manual Capture end
                self.g.capture_end()

        def training_step(self, batch, batch_idx):
            """Automatic optimization training step."""
            if batch_idx < 2:
                # Regular training for graph capture
                x, y = batch
                loss = f.cross_entropy(self.forward(x), y)
                self.log("train_loss", loss)
                return loss
            # Replay the captured graph
            data, target = batch
            self.static_input.copy_(data)
            self.static_target.copy_(target)
            self.g.replay()
            self.log("train_loss", self.static_loss)
            return self.static_loss

    loss_metrics = []
    # Train with baseline model
    seed_everything(42)
    model = get_model(HPUGraphMode.TRAIN_NONE)
    data_module = MNISTDataModule(batch_size=200)
    loss_metrics.append(train_model(tmpdir, 1, model=model, data_module=data_module, profiler=None))

    # Train with HPU Graphs
    seed_everything(42)
    hpu_graph_model = ManualCaptureModel()
    data_module = MNISTDataModule(batch_size=200)
    loss_metrics.append(train_model(tmpdir, 1, model=hpu_graph_model, data_module=data_module, profiler=None))

    assert torch.allclose(
        loss_metrics[0]["train_loss"], loss_metrics[1]["train_loss"], rtol=0.05
    ), loss_metrics  # Compare train loss
    assert torch.allclose(
        loss_metrics[0]["val_acc"], loss_metrics[1]["val_acc"], rtol=0.01
    ), loss_metrics  # Compare val acc
