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

import csv
import os

import pytest
import torch
from habana_frameworks.torch.hpu.metrics import metric_global
from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model
from lightning_habana import HPUAccelerator, HPUDDPStrategy, SingleHPUStrategy
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel


class DynamicOpsBoringModel(BoringModel):
    """Boring model with dynamicity."""

    def forward(self, x):
        x = super().forward(x)
        # Dynamic op: boolean indexing
        x = x[x < 0]
        x = x.view(-1, 1)
        return torch.mm(x, torch.eye(x.shape[1], device=x.device, dtype=x.dtype))


def run_training(tmpdir, hpus, model, data_module):
    """Init trainer and run fit."""
    seed_everything(42)
    gc_metric = metric_global("graph_compilation")
    model = model()
    _strategy = HPUDDPStrategy() if hpus > 1 else SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=_strategy,
        max_epochs=1,
        fast_dev_run=5,
    )
    trainer.fit(model, data_module())
    return gc_metric.stats()


def test_dynamic_shapes_recompilations_recipe_caching(tmpdir, hpus, monkeypatch):
    """Tests number of recompilations between cached and non-cached runs."""
    default_compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)

    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", f"{tmpdir}/recipes,True,1024")
    cached_compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)

    assert cached_compiles[0] <= default_compiles[0]


def test_dynamic_shapes_graph_compiler(tmpdir, hpus, monkeypatch):
    """Test number of recompilations with GC support for dynamic shapes."""
    default_compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)

    monkeypatch.setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1")
    cached_compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)

    assert cached_compiles[0] <= default_compiles[0]

@pytest.mark.standalone_only()
def test_dynamic_shapes_auto_detect_recompilations(tmpdir):
    """Test auto_detect_recompilations tool."""

    def calculate_auto_detect_total_recompile_counts(csv_file_path):
        total_recompile_counts = 0

        try:
            with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    recompile_count = int(row["Recompile count"])
                    total_recompile_counts += recompile_count
            return total_recompile_counts
        except FileNotFoundError:
            print(f"Error: CSV file not found: {csv_file_path}")
            return None

    seed_everything(42)
    model = DynamicOpsBoringModel()
    net = detect_recompilation_auto_model(model, csv_out=os.path.join(tmpdir, "out.csv"))
    data_module = BoringDataModule
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=1,
        strategy=SingleHPUStrategy(),
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=0,
    )
    trainer.fit(net, data_module())
    # This dumps two csv:
    # 1. filename.csv_1.csv: This has details of recompiles
    # 2. filename.csv_2.csv: This has number of recompiles per module
    net.analyse_dynamicity()
    recompiles = calculate_auto_detect_total_recompile_counts(os.path.join(tmpdir, "out.csv_2.csv"))
    assert recompiles is not None
