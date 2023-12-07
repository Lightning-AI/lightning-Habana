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

import torch.multiprocessing as mp
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

DEFAULT_RUN_TYPE = [
    "basic",
    "autocast",
    "HPUPrecisionPlugin",
    "MixedPrecisionPlugin",
    "recipe_caching",
]

OPTIONAL_RUN_TYPE = [
    "multi_tenancy",
]


def parse_args():
    """Cmdline arguments parser."""
    parser = argparse.ArgumentParser(description="Example to showcase features when training on HPU.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    parser.add_argument(
        "-r",
        "--run_types",
        nargs="+",
        choices=DEFAULT_RUN_TYPE + OPTIONAL_RUN_TYPE,
        default=DEFAULT_RUN_TYPE,
        help="Select run type for example",
    )
    parser.add_argument("-n", "--num_tenants", type=int, default=2, help="Number of tenants to run on node")
    parser.add_argument("-c", "--devices_per_tenant", type=int, default=2, help="Number of devices per tenant")
    parser.add_argument("-d", "--devices", type=int, default=1, help="Number of devices for basic runs")
    return parser.parse_args()


def run_trainer(model, data_module, run_type, plugin=[], devices=1):
    """Run trainer.fit and trainer.test with given parameters."""
    _devices = devices
    _strategy = HPUParallelStrategy(start_method="spawn") if _devices > 1 else SingleHPUStrategy()
    if run_type == "recipe_caching":
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = "tmp/recipes,True,1024"
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=_devices,
        strategy=_strategy,
        plugins=plugin,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)
    if run_type == "recipe_caching":
        os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)


def spawn_tenants(model, data_module, run_type, devices, num_tenants):
    """Spawn multiple WL tenants on a single node."""
    processes = []
    for _ in range(num_tenants):
        processes.append(mp.Process(target=run_trainer, args=(model, data_module, [], run_type, devices)))

    for tenant, process in enumerate(processes):
        modules = ",".join(str(i) for i in range(tenant * devices, (tenant + 1) * devices))
        # Cannot dynamically check for port availability as main launches all the processes
        # before the launched process can bind the ports.
        # So check for free port on any given port always returns True
        port = 1234 + tenant
        os.environ["HABANA_VISIBLE_MODULES"] = str(modules)
        os.environ["MASTER_PORT"] = str(port)
        print(f"Spawning {tenant=} with {modules=}, and {port=}")
        process.start()
        os.environ.pop("HABANA_VISIBLE_MODULES", None)
        os.environ.pop("MASTER_PORT", None)

    for process in processes:
        process.join()


def init_model_and_plugins(run_type, options):
    """Picks appropriate model and plugin."""
    # Defaults
    model = LitClassifier()
    data_module = MNISTDataModule(batch_size=32)

    # Init model and data_module
    if run_type == "autocast":
        if "LOWER_LIST" in os.environ or "FP32_LIST" in os.environ:
            model = LitAutocastClassifier(op_override=True)
        else:
            model = LitAutocastClassifier()
        warnings.warn(
            "To override operators with autocast, set LOWER_LIST and FP32_LIST file paths as env variables."
            "Example: LOWER_LIST=<path_to_bf16_ops> python example.py"
            "https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/Autocast.html#override-options"
        )

    # Add plugins here
    plugin = []
    if run_type == "HPUPrecisionPlugin":
        plugin = [HPUPrecisionPlugin(device="hpu", precision="bf16-mixed")]
    elif run_type == "MixedPrecisionPlugin":
        plugin = [MixedPrecisionPlugin(device="hpu", precision="bf16-mixed")]

    if options.verbose:
        print(f"Running {model=} with {run_type=} and {plugin=}")

    devices_per_tenant = options.devices_per_tenant
    devices = options.devices
    if run_type == "multi_tenancy":
        max_devices = 8
        num_tenants = options.num_tenants
        if max_devices < num_tenants * devices_per_tenant:
            devices_per_tenant = max_devices if devices_per_tenant > max_devices else devices_per_tenant
            num_tenants = max_devices // devices_per_tenant
            warnings.warn(
                f"""More cards requested than available.
                Launching {num_tenants} tenants, with {devices_per_tenant} cards each"""
            )
        if options.verbose:
            print(f"Running with {num_tenants} tenants, using {devices_per_tenant} cards per tenant")
        spawn_tenants(model, data_module, run_type, devices_per_tenant, num_tenants)
    else:
        run_trainer(model, data_module, run_type, plugin, devices)


if __name__ == "__main__":
    # Get options
    options = parse_args()
    if options.verbose:
        print(f"Running MNIST mixed precision training with {options=}")
    # Run model and print accuracy
    for _run_type in options.run_types:
        seed_everything(42)
        init_model_and_plugins(_run_type, options)
