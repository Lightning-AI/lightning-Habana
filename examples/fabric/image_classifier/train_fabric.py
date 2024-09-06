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
"""Here are the steps to use Fabric in your PyTorch code with HPU.

1. Create the Lightning Fabric object at the beginning of your script.

2. Use HPU accelerator and strategy while creating the Fabric object

3. Run the script from the terminal using ``python model path/to/train.py``

"""

import argparse
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F  # NOQA
import torch.optim as optim
import torchvision.transforms as T  # NOQA
from lightning.fabric import Fabric, seed_everything
from lightning_habana.fabric.accelerator import HPUAccelerator
from lightning_habana.fabric.strategies.single import SingleHPUStrategy
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "..", "Datasets")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Driver method
def run(hparams):
    """Driver method to run the training loop."""
    fabric = Fabric(
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
    )
    seed_everything(hparams.seed)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    with fabric.rank_zero_first(local=False):
        train_dataset = MNIST(DATASETS_PATH, download=fabric.is_global_zero, train=True, transform=transform)
        test_dataset = MNIST(DATASETS_PATH, download=fabric.is_global_zero, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size)

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=hparams.lr)

    model, optimizer = fabric.setup(model, optimizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

    test_acc = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    for epoch in range(1, hparams.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            fabric.backward(loss)

            optimizer.step()
            if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
                if hparams.dry_run:
                    break

        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()

                test_acc(output, target)

                if hparams.dry_run:
                    break

        test_loss = fabric.all_gather(test_loss).sum() / len(test_loader.dataset)

        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({100 * test_acc.compute():.0f}%)\n")
        test_acc.reset()

        if hparams.dry_run:
            break

    if hparams.save_model:
        fabric.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    hparams = parser.parse_args()

    run(hparams)
