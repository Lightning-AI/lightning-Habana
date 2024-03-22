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
from contextlib import contextmanager, suppress

import torch.multiprocessing as mp
from lightning_utilities import module_available
from mnist_sample import LitAutocastClassifier, LitClassifier

if module_available("lightning"):
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
    from lightning.pytorch.plugins.precision import MixedPrecision
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule
    from pytorch_lightning.plugins.precision import MixedPrecision

from lightning_habana import HPUAccelerator, HPUDDPStrategy, HPUPrecisionPlugin, SingleHPUStrategy

DEFAULT_RUN_TYPE = [
    "basic",
    "autocast",
    "HPUPrecisionPlugin",
    "MixedPrecisionPlugin",
    "recipe_caching",
    "fp8_training",
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


@contextmanager
def set_env_vars(env_dict):
    """CM to set env."""
    active = bool(env_dict)  # Set to True if env_dict is not None or empty
    original_values = {key: os.environ.get(key, None) for key in env_dict} if active else {}

    with suppress(Exception):
        if active:
            os.environ.update(env_dict)
        yield

    # Restore env
    for key, original_value in original_values.items():
        if original_value is not None:
            os.environ[key] = original_value
        else:
            del os.environ[key]


def run_trainer(model, data_module, plugin, devices=1, strategy=None):
    """Run trainer.fit with given parameters."""
    if strategy is None:
        strategy = HPUDDPStrategy() if devices > 1 else SingleHPUStrategy()
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=devices,
        strategy=strategy,
        plugins=plugin,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)


def spawn_tenants(model, data_module, devices, num_tenants):
    """Spawn multiple WL tenants on a single node."""
    processes = []
    for _ in range(num_tenants):
        processes.append(
            mp.Process(
                target=run_trainer, args=(model, data_module, None, devices, HPUDDPStrategy(start_method="spawn"))
            )
        )

    for tenant, process in enumerate(processes):
        modules = ",".join(str(i) for i in range(tenant * devices, (tenant + 1) * devices))
        # Cannot dynamically check for port availability as main launches all the processes
        # before the launched process can bind the ports.
        # So check for free port on any given port always returns True
        port = 1234 + tenant
        env_dict = {
            "HABANA_VISIBLE_MODULES": str(modules),
            "MASTER_PORT": str(port),
        }
        with set_env_vars(env_dict):
            print(f"Spawning {tenant=} with {modules=}, and {port=}")
            process.start()

    for process in processes:
        process.join()


def get_model(run_type):
    """Pick model."""
    # Default
    model = LitClassifier()
    # WA for https://github.com/Lightning-AI/pytorch-lightning/issues/4450
    data_module = MNISTDataModule(batch_size=32, num_workers=1)

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
    return model, data_module


def get_plugins(run_type):
    """Select plugin."""
    if run_type == "HPUPrecisionPlugin":
        return HPUPrecisionPlugin(device="hpu", precision="bf16-mixed")
    if run_type == "MixedPrecisionPlugin":
        return MixedPrecision(device="hpu", precision="bf16-mixed")
    if run_type == "fp8_training":
        return HPUPrecisionPlugin(device="hpu", precision="fp8")
    return None


def run_training(run_type, options, model, data_module, plugin):
    """Run training.

    Picks between regular training or launching multiple tenants per node.

    """
    if run_type == "multi_tenancy":
        devices_per_tenant = options.devices_per_tenant
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
        spawn_tenants(model, data_module, devices_per_tenant, num_tenants)
    else:
        env_dict = {"PT_HPU_RECIPE_CACHE_CONFIG": "tmp/recipes,True,1024"} if run_type == "recipe_caching" else None
        with set_env_vars(env_dict):
            run_trainer(model, data_module, plugin, options.devices)


if __name__ == "__main__":
    # Get options
    options = parse_args()
    if options.verbose:
        print(f"Running MNIST mixed precision training with {options=}")
    # Run model and print accuracy
    for _run_type in options.run_types:
        if _run_type == "fp8_training" and HPUAccelerator.get_device_name() == "GAUDI":
            print("fp8 training not supported on GAUDI. Skipping.")
            continue

        seed_everything(42)
        model, data_module = get_model(_run_type)
        plugin = get_plugins(_run_type)
        if _run_type == "fp8_training":
            plugin.convert_modules(model)

        if options.verbose:
            print(f"Running {_run_type=} with {model=}, and {plugin=}")
        run_training(_run_type, options, model, data_module, plugin)
