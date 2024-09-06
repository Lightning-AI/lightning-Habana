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

import argparse
import random

import habana_frameworks.torch.core as htcore
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

from lightning_habana.pytorch import HPUAccelerator, HPUDDPStrategy, SingleHPUStrategy


def parse_args():
    """Arguments Parser."""
    parser = argparse.ArgumentParser(description="Example for using HPU Graphs with PyTorch Lightning models.")

    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbosity")

    subparsers = parser.add_subparsers(help="run type help", dest="run_type")
    subparser_train = subparsers.add_parser("train", help="Show usage of HPU graphs for training")
    subparser_train.add_argument(
        "--mode",
        choices=["capture_and_replay", "make_graphed_callables", "modulecacher"],
        nargs="+",
        help="Methods for training",
    )

    subparser_inference = subparsers.add_parser("inference", help="Show usage of HPU graphs for inference.")
    subparser_inference.add_argument(
        "--mode", choices=["capture_and_replay", "wrap_in_hpu_graph"], nargs="+", help="Methods for inference"
    )

    subparser_inference = subparsers.add_parser("dynamicity", help="Show usage of HPU graphs for inference.")
    subparser_inference.add_argument(
        "--mode", choices=["dynamic_control_flow", "dynamic_ops"], nargs="+", help="Methods to tackle dynamicity"
    )
    return parser.parse_args()


class NetHPUGraphs(LightningModule):
    """Model with modifications to support HPU graphs."""

    def __init__(self, mode=None, batch_size=None):
        """Init."""
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.loss = []
        self.mode = mode

        # Define modes to run the example
        if self.mode == "dynamic_ops":
            # Encapsulate dynamic ops between two separate HPU Graph modules,
            # instead of using one single HPU Graph for whole model
            self.module1 = NetHPUGraphs()
            self.module2 = nn.Identity()
            htcore.hpu.ModuleCacher(max_graphs=10)(model=self.module1, inplace=True)
            htcore.hpu.ModuleCacher(max_graphs=10)(model=self.module2, inplace=True)
            self.automatic_optimization = False
            self.training_step = self.dynamic_ops_training_step
        elif self.mode == "dynamic_control_flow":
            # Break Model into separate HPU Graphs for each control flow.
            self.module1 = NetHPUGraphs()
            self.module2 = nn.Identity()
            self.module3 = nn.ReLU()
            htcore.hpu.ModuleCacher(max_graphs=10)(model=self.module1, inplace=True)
            htcore.hpu.ModuleCacher(max_graphs=10)(model=self.module2, inplace=True)
            htcore.hpu.ModuleCacher(max_graphs=10)(model=self.module3, inplace=True)
            self.automatic_optimization = False
            self.training_step = self.dynamic_control_flow_training_step
        elif self.mode == "capture_and_replay":
            self.g = htcore.hpu.HPUGraph()
            self.automatic_optimization = False
            self.training_step = self.train_with_capture_and_replay
            self.static_input = torch.randn((batch_size), 1, 28, 28, device="hpu")
            self.static_target = torch.randint(0, 10, (batch_size,), device="hpu")
            self.static_y_pred = torch.randint(0, 10, (batch_size,), device="hpu")
            self.static_loss = None
        else:
            self.training_step = self.training_step_automatic

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

        # Finally the main training loop
        # data must be copied to existing tensors that were used in the capture phase
        data, target = batch
        self.static_input.copy_(data)
        self.static_target.copy_(target)
        self.g.replay()
        self.log("train_loss", self.static_loss)
        # result is available in static_loss tensor after graph is replayed
        return self.static_loss

    def dynamic_ops_training_step(self, batch, batch_idx):
        """Training step with HPU Graphs and Dynamic ops."""
        optimizer = self.optimizers()
        data, target = batch
        optimizer.zero_grad(set_to_none=True)
        # Train with HPU graph module
        tmp = self.module1(torch.flatten(data, 1))

        # Dynamic op
        htcore.mark_step()
        tmp = tmp[torch.where(tmp < 0)]
        htcore.mark_step()

        # Resume training with HPU graph module
        tmp = self.module2(tmp)
        loss = f.cross_entropy(torch.reshape(tmp, (200, 10)), target)
        loss.backward()
        optimizer.step()
        self.log("train_loss", loss)
        return loss

    def dynamic_control_flow_training_step(self, batch, batch_idx):
        """Training step with HPU Graphs and Dynamic control flow."""
        optimizer = self.optimizers()
        data, target = batch
        optimizer.zero_grad(set_to_none=True)
        # Train with HPU Graph
        tmp = self.module1(torch.flatten(data, 1))

        # dynamic control flow
        # forward ops run as a graph
        tmp = self.module2(tmp) if random.random() > 0.5 else self.module3(tmp)

        loss = f.cross_entropy(torch.reshape(tmp, (200, 10)), target)
        loss.backward()
        optimizer.step()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        if self.mode == "capture_and_replay":
            if batch_idx == 0:
                self.g.capture_begin()
                static_y_pred = self.forward(self.static_input)
                self.static_loss = f.cross_entropy(static_y_pred, self.static_target)
                self.g.capture_end()
            else:
                htcore.mark_step()
                self.static_input.copy_(x)
                self.static_target.copy_(y)
                self.g.replay()
            acc = self.accuracy(None, y, self.static_y_pred)
        else:
            logits = self.forward(x)
            acc = self.accuracy(logits, y)
        self.log("test_acc", acc)

    @staticmethod
    def accuracy(logits, y, pred=None):
        """Calculate accuracy."""
        if pred is not None:
            return torch.sum(torch.eq(pred, y).to(torch.float32)) / len(y)
        return torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=0.1)


def train_model(model, data_module, hpus=1, mode="fit"):
    """Runs trainer.<fit / validate>."""
    _strategy = SingleHPUStrategy()
    if hpus > 1:
        _strategy = HPUDDPStrategy()
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=_strategy,
        max_epochs=1,
        fast_dev_run=3,
    )
    print(f"starting {mode}")
    if hasattr(trainer, mode):
        func = getattr(trainer, mode)
    func(model, data_module)
    return trainer.logged_metrics


def get_model(run_type, mode):
    """Returns model instances depending on HPU graph mode."""
    if mode is None:
        return NetHPUGraphs(mode=mode)

    if run_type == "train":
        if mode == "capture_and_replay":
            # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#training-loop-with-capture-and-replay
            return NetHPUGraphs(mode=mode, batch_size=200)
        if mode == "make_graphed_callables":
            # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#training-loop-with-make-graphed-callables
            model = NetHPUGraphs(mode=mode).to(torch.device("hpu"))
            x = torch.randn(200, 1, 28, 28, device="hpu")
            return htcore.hpu.make_graphed_callables(model, (x,))
        if mode == "modulecacher":
            # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#training-loop-with-modulecacher
            model = NetHPUGraphs(mode=mode)
            return htcore.hpu.ModuleCacher(max_graphs=10)(model=model, inplace=True)

    elif run_type == "inference":
        if mode == "capture_and_replay":
            # Inference: https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html
            return NetHPUGraphs(mode=mode, batch_size=200)
        if mode == "wrap_in_hpu_graph":
            # https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html
            model = NetHPUGraphs(mode=mode).to(torch.device("hpu"))
            return htcore.hpu.wrap_in_hpu_graph(model, asynchronous=False, disable_tensor_cache=True)

    elif run_type == "dynamicity":
        # https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/HPU_Graphs_Training.html#dynamicity-in-models
        return NetHPUGraphs(mode=mode)
    return None


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        print(f"{args=}")

    _train_type = (
        "fit"
        if (args.run_type == "train" or args.run_type == "dynamicity")
        else "test"
        if args.run_type == "inference"
        else None
    )

    for mode in args.mode:
        if args.verbose:
            print(f"{mode=}")
        seed_everything(42)
        _model = get_model(args.run_type, mode)
        assert _model is not None
        _data_module = MNISTDataModule(batch_size=200)

        loss_metrics = train_model(hpus=1, model=_model, data_module=_data_module, mode=_train_type)
        print(f"{loss_metrics}")
