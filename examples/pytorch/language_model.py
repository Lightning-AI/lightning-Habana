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

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning_utilities import module_available
from lightning.pytorch.utilities import rank_zero_info


if module_available("lightning"):
    from lightning.pytorch import Trainer, seed_everything, LightningModule
    from lightning.pytorch.demos import Transformer, WikiText2
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, seed_everything, LightningModule
    from pytorch_lightning.demos import Transformer, WikiText2


from lightning_habana.pytorch.strategies import HPUFSDPStrategy, HPUDDPStrategy
from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins import HPUFSDPPrecision, HPUPrecisionPlugin
import habana_frameworks.torch as htorch

class LanguageModel(LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(
            vocab_size=vocab_size,
            nlayers=32,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


def parse_args():
    """Cmdline arguments parser."""
    parser = argparse.ArgumentParser(description="Example to showcase features when training on HPU.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    parser.add_argument("-d", "--devices", type=int, default=2, help="Number of devices for basic runs")
    parser.add_argument("-s", "--strategy", type=str, default="FULL_SHARD", help="FSDP strategies to be applied.\
                                                                 The value can be one of FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD")

    return parser.parse_args()


if __name__ == "__main__":

    seed_everything(42)

    options = parse_args()
    if options.verbose:
        print(f"Running language model with FSDP on HPU with {options=}")

    if options.devices < 2:
        print(f"The script requires a multi device setup")
        sys.exit(1)

    if options.strategy not in ["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD"]:
        print(f"Invalid strategy {options.strategy}. The value must be one of FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD")
        sys.exit(1)

    policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset)

    model = LanguageModel(vocab_size=dataset.vocab_size)
    # strategy = HPUDDPStrategy()
    # plugin=HPUPrecisionPlugin(device="hpu", precision="bf16-mixed")
    # trainer = Trainer(
    #     accelerator=HPUAccelerator(),
    #     devices=8,
    #     strategy=strategy,
    #     plugins=plugin,
    #     fast_dev_run=10,
    #     enable_model_summary=True
    # )
    # trainer.fit(model, train_dataloader)
    # rank_zero_info(f"Peak Memory alloc using DDP on HPU: {htorch.hpu.max_memory_allocated() / (1024**3)} GB")

    htorch.hpu.reset_peak_memory_stats()
    model = LanguageModel(vocab_size=dataset.vocab_size)
    _strategy=HPUFSDPStrategy(parallel_devices=[torch.device("hpu")] * options.devices,
                                sharding_strategy=options.strategy,
                                auto_wrap_policy=policy,
                                precision_plugin=HPUFSDPPrecision("bf16-mixed")
                            )

    trainer = Trainer(accelerator=HPUAccelerator(), strategy=_strategy, fast_dev_run=10, enable_model_summary=True)
    trainer.fit(model, train_dataloader)
    rank_zero_info(f"Peak Memory alloc using FSDP on HPU: {htorch.hpu.max_memory_allocated() / (1024**3)} GB")


