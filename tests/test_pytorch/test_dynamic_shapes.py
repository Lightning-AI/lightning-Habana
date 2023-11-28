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
import json
import os

import torch
from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model
from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy
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


def calculate_auto_detect_total_recompile_counts(csv_file_path):
    total_recompile_counts = 0

    try:
        with open(csv_file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                recompile_count = int(row["Recompile count"])
                total_recompile_counts += recompile_count
        return total_recompile_counts
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return None


def run_training(tmpdir, hpus, model, data_module):
    """Init trainer and run fit."""
    seed_everything(42)
    model = model()
    _strategy = HPUParallelStrategy() if hpus > 1 else SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=hpus,
        strategy=_strategy,
        max_epochs=1,
        fast_dev_run=5,
    )
    trainer.fit(model, data_module())
    return get_metric_compiles(tmpdir)


def get_metric_compiles(dirpath):
    """Reads metric files and returns recompiles for each trainer rank."""
    json_files = [file for file in os.listdir(dirpath) if ".json" in file]
    compiles = {}
    assert json_files is not []
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


def test_dynamic_shapes_metric_file_dump(tmpdir, hpus):
    """Tests metric file is generated."""
    compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)
    assert compiles is not {}


def test_dynamic_shape_recompilations_recipe_caching(tmpdir, hpus):
    """Tests number of recompilations between cached and non-cached runs."""
    base_path = f"{tmpdir}/base"
    compiled_path = f"{tmpdir}/compiled"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(compiled_path):
        os.mkdir(compiled_path)

    os.environ["PT_HPU_METRICS_FILE"] = os.path.join(base_path, "metrics.json")
    os.environ["PT_HPU_METRICS_DUMP_TRIGGERS"] = "process_exit,metric_change"
    default_compiles = run_training(base_path, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)

    os.environ["PT_HPU_METRICS_FILE"] = os.path.join(compiled_path, "metrics.json")
    os.environ["PT_HPU_METRICS_DUMP_TRIGGERS"] = "process_exit,metric_change"
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{tmpdir}/recipes,True,1024"
    cached_compiles = run_training(compiled_path, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)
    os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)

    for key, value in cached_compiles.items():
        assert key in default_compiles
        assert value <= default_compiles[key]


def test_dynamic_shapes_graph_compiler(tmpdir, hpus):
    """Test number of recompilations with GC support for dynamic shapes."""
    default_compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)

    os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"
    cached_compiles = run_training(tmpdir, hpus=hpus, model=DynamicOpsBoringModel, data_module=BoringDataModule)
    del os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"]

    for key, value in cached_compiles.items():
        assert key in default_compiles
        assert value <= default_compiles[key]


def test_dynamic_shapes_auto_detect_recompilations(tmpdir):
    """Test auto_detect_recompilations tool."""
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
