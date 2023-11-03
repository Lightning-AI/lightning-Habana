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

import json
import os

import pytest
import torch
import torch.multiprocessing as mp
from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy
from lightning_habana.utils.resources import device_count
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel


class DynamicOpsBoringModel(BoringModel):
    """Boring model with dynamicity."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 16)

    def forward(self, x):
        x = super().forward(x)
        # Dynamic op: boolean indexing
        x = x[x < 0]
        x = x.view(-1, 1)
        return torch.mm(x, torch.eye(x.shape[1], device=x.device, dtype=x.dtype))

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("train_loss", loss["loss"])
        return loss


def run_training(tmpdir, hpus, model, data_module, metric_file_path, return_dict):
    """Init trainer and run fit."""
    seed_everything(42)
    _strategy = HPUParallelStrategy(start_method="spawn") if hpus > 1 else SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=_strategy,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=0,
    )
    trainer.fit(model(), data_module())
    if trainer.global_rank == 0:
        compiles = get_compiles(metric_file_path)
        return_dict.update(compiles)
    trainer.strategy.barrier()


def get_compiles(dirpath):
    """Reads metric files and returns recompiles for each trainer rank."""
    json_files = [file for file in os.listdir(dirpath) if ".json" in file]
    compiles = {}
    for json_file in json_files:
        max_compiles = 0
        with open(os.path.join(dirpath, json_file)) as file:
            json_content = file.read()
            # Check if the last character is ']' to determine if json is complete
            if json_content.strip()[-1] != "]":
                json_content += "]"
            data = json.loads(json_content)
            for entry in data:
                if entry.get("metric_name") == "graph_compilation":
                    statistics = entry.get("statistics")
                    if statistics:
                        total_number = statistics.get("TotalNumber")
                        if total_number is not None and total_number > max_compiles:
                            max_compiles = total_number
        os.remove(os.path.join(dirpath, json_file))
        compiles[json_file] = max_compiles
    return compiles


@pytest.mark.skipif(device_count() < 2, reason="Test requires more than 1 HPU.")
def test_dynamic_shape_recompilations_recipe_caching(tmpdir):
    """Tests number of recompilations between cached and non-cached runs."""
    # Set env for metric files
    metric_file_path = os.path.join(tmpdir, "metrics")
    os.environ["PT_HPU_METRICS_FILE"] = os.path.join(metric_file_path, "metrics.json")
    os.environ["PT_HPU_METRICS_DUMP_TRIGGERS"] = "process_exit,metric_change"

    _model = DynamicOpsBoringModel
    _data_module = BoringDataModule

    # Run with recipe caching disabled
    manager = mp.Manager()
    m1 = manager.dict()
    p1 = mp.Process(target=run_training, args=(tmpdir, 2, _model, _data_module, metric_file_path, m1))
    p1.start()
    p1.join()
    assert p1.exitcode == 0

    # Run with recipe caching enabled
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{tmpdir}/recipes,True,1024"
    m2 = manager.dict()
    p2 = mp.Process(target=run_training, args=(tmpdir, 2, _model, _data_module, metric_file_path, m2))

    p2.start()
    p2.join()
    assert p2.exitcode == 0
    os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)

    for key in m1:
        assert key in m2
        assert m1[key] >= m2[key], "More compiles in cached run"

    # Unset env
    os.environ.pop("PT_HPU_METRICS_FILE", None)
    os.environ.pop("PT_HPU_METRICS_DUMP_TRIGGERS", None)
