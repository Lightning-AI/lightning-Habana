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

import argparse
import os
import sys

import torch
import torch.nn as nn
from lightning_utilities import module_available
from torch.utils.data import DataLoader

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer, seed_everything
    from lightning.pytorch.demos import Transformer, WikiText2
    from lightning.pytorch.utilities import rank_zero_info
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.demos import Transformer, WikiText2
    from pytorch_lightning.utilities import rank_zero_info


import habana_frameworks.torch as htorch
from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUFSDPPrecision, HPUPrecisionPlugin
from lightning_habana.pytorch.strategies import HPUDDPStrategy, HPUFSDPStrategy
from lightning_habana.utils.imports import _LIGHTNING_GREATER_EQUAL_2_3_0
from lightning_habana.utils.resources import is_hpu_initialized


class LanguageModel(LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(
            vocab_size=vocab_size,
        )

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = nn.functional.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


def parse_args():
    """Cmdline arguments parser."""
    parser = argparse.ArgumentParser(description="Example to showcase features when training on HPU.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    parser.add_argument("-d", "--devices", type=int, default=2, help="Number of devices for basic runs")
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="FULL_SHARD",
        help="FSDP/DDP strategies to be applied.\
            The value can be one of FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD, DDP",
    )

    return parser.parse_args()


if __name__ == "__main__":

    seed_everything(42)

    options = parse_args()
    if not _LIGHTNING_GREATER_EQUAL_2_3_0:
        print("The example requires lightning version 2.3.0 or above")
        exit(0)

    if options.verbose:
        print(f"Running language model with FSDP on HPU with {options=}")

    os.environ["PT_HPU_LAZY_MODE"] = "0"

    if options.devices < 2:
        print("The script requires a multi device setup")
        sys.exit(1)

    if options.strategy not in ["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD", "DDP"]:
        print(
            f"Invalid strategy {options.strategy}.\
            The value must be one of FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD"
        )
        sys.exit(1)

    policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset)

    if options.strategy == "DDP":
        model = LanguageModel(vocab_size=dataset.vocab_size)
        strategy = HPUDDPStrategy(parallel_devices=[torch.device("hpu")] * options.devices)
        plugin = HPUPrecisionPlugin(device="hpu", precision="bf16-mixed")
        trainer = Trainer(
            accelerator=HPUAccelerator(),
            devices=options.devices,
            strategy=strategy,
            plugins=plugin,
            fast_dev_run=10,
            enable_model_summary=True,
        )
        trainer.fit(model, train_dataloader)
        rank_zero_info(
            f"Peak Memory alloc using DDP strategy on HPU: {htorch.hpu.max_memory_allocated() / (1024**3)} GB"
        )
    else:
        if is_hpu_initialized():
            htorch.hpu.reset_peak_memory_stats()
        model = LanguageModel(vocab_size=dataset.vocab_size)
        _strategy = HPUFSDPStrategy(
            parallel_devices=[torch.device("hpu"), torch.hpu.current_device()] * options.devices,
            sharding_strategy=options.strategy,
            auto_wrap_policy=policy,
            precision_plugin=HPUFSDPPrecision("bf16-mixed"),
        )

        trainer = Trainer(accelerator=HPUAccelerator(), strategy=_strategy, fast_dev_run=1, enable_model_summary=True)
        trainer.fit(model, train_dataloader)
        rank_zero_info(
            f"Peak Memory alloc using FSDP {options.strategy} strategy "
            f" on HPU: {htorch.hpu.max_memory_allocated() / (1024**3)} GB"
        )
