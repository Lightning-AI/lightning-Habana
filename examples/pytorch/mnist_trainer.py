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
import warnings

from lightning_utilities import module_available
from mnist_sample import LitAutocastClassifier, LitClassifier

if module_available("lightning"):
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
    from lightning.pytorch.plugins.precision import MixedPrecisionPlugin
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule
    from pytorch_lightning.plugins.precision import MixedPrecisionPlugin

from lightning_habana import HPUAccelerator, HPUParallelStrategy, HPUPrecisionPlugin, SingleHPUStrategy
from lightning_habana.utils.resources import device_count

RUN_TYPE = ["basic", "autocast", "recipe_caching"]
PLUGINS = ["None", "HPUPrecisionPlugin", "MixedPrecisionPlugin"]


def parse_args():
    """Cmdline arguments parser."""
    parser = argparse.ArgumentParser(description="Example to showcase mixed precision training with HPU.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    parser.add_argument(
        "-r", "--run_types", nargs="+", choices=RUN_TYPE, default=RUN_TYPE, help="Select run type for example"
    )
    parser.add_argument(
        "-p",
        "--plugins",
        nargs="+",
        default=PLUGINS,
        choices=PLUGINS,
        help="Plugins for use in training",
    )
    return parser.parse_args()


def run_trainer(model, plugin, run_type):
    """Run trainer.fit and trainer.test with given parameters."""
    _data_module = MNISTDataModule(batch_size=32)
    _devices = 1
    _strategy = SingleHPUStrategy()
    if run_type == "recipe_caching":
        if device_count() < 2:
            warnings.warn("Recipe caching not required for single device.")
        else:
            _devices = 2
            _strategy = HPUParallelStrategy(
                recipe_cache_path="/tmp/recipes",
                recipe_cache_clean=True,
                recipe_cache_size=1024,
            )
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=_devices,
        strategy=_strategy,
        plugins=plugin,
        fast_dev_run=3,
    )
    trainer.fit(model, _data_module)
    trainer.test(model, _data_module)


def check_and_init_plugin(plugin, verbose):
    """Initialise plugins with appropriate checks."""
    if verbose:
        print(f"Initializing {plugin}")
    if plugin == "HPUPrecisionPlugin":
        return [HPUPrecisionPlugin(device="hpu", precision="bf16-mixed")]
    if plugin == "MixedPrecisionPlugin":
        warnings.warn("Operator overriding is not supported with MixedPrecisionPlugin on Habana devices.")
        return [MixedPrecisionPlugin(device="hpu", precision="bf16-mixed")]
    return []


def run_model(run_type, plugin, verbose):
    """Picks appropriate model and plugin."""
    _model = LitClassifier()
    if run_type == "autocast":
        if plugin is not None:
            warnings.warn("Skipping precision plugins. Redundant with autocast run.")
        if "LOWER_LIST" in os.environ or "FP32_LIST" in os.environ:
            _model = LitAutocastClassifier(op_override=True)
        else:
            _model = LitAutocastClassifier()
        warnings.warn(
            "To override operators with autocast, set LOWER_LIST and FP32_LIST file paths as env variables."
            "Example: LOWER_LIST=<path_to_bf16_ops> python example.py"
            "https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/Autocast.html#override-options"
        )

    _plugin = check_and_init_plugin(plugin, verbose) if plugin != "None" else None
    if verbose:
        print(f"With run type: {run_type}, running model: {_model} with plugin: {plugin}")
    run_trainer(_model, _plugin, run_type)


if __name__ == "__main__":
    # Get options
    options = parse_args()
    if options.verbose:
        print(f"Running MNIST mixed precision training with options: {options}")

    # Run model and print accuracy
    for _run_type in options.run_types:
        for plugin in options.plugins:
            seed_everything(42)
            run_model(_run_type, plugin, options.verbose)
