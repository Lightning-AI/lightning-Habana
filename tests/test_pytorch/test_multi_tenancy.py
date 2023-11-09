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


import os

import pytest
import torch.multiprocessing as mp
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy


def run_train(tmpdir, _devices, tenant, status):
    seed_everything(42)
    _model = BoringModel()
    _data_module = BoringDataModule()
    _strategy = HPUParallelStrategy(start_method="spawn") if _devices > 1 else SingleHPUStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=HPUAccelerator(),
        devices=_devices,
        strategy=_strategy,
        fast_dev_run=3,
    )
    try:
        trainer.fit(_model, _data_module)
        status[tenant] = None
    except Exception as e:
        status[tenant] = str(e)


def spawn_tenants(tmpdir, _num_tenants, _cards_per_tenant):
    processes = []
    manager = mp.Manager()
    status = manager.dict()
    for i in range(_num_tenants):
        processes.append(mp.Process(target=run_train, args=(tmpdir, _cards_per_tenant, i, status)))

    for tenant, process in enumerate(processes):
        modules = ",".join(str(i) for i in range(tenant * _cards_per_tenant, (tenant + 1) * _cards_per_tenant))
        # Cannot dynamically check for port availability as main launches all the processes
        # before the launched process can bind the ports.
        # So check for free port on any given port always returns True
        port = 12345 + tenant * 10
        custom_env = {"HABANA_VISIBLE_MODULES": str(modules), "MASTER_PORT": str(port)}
        os.environ.update(custom_env)
        process.start()

    for process in processes:
        process.join()

    return status


def test_multi_tenancy_more_cards_than_visible(tmpdir):
    expected = """AssertionError: There is not enough devices
        available for training. Please verify if HABANA_VISIBLE_MODULES
        is set correctly"""
    os.environ["HABANA_VISIBLE_MODULES"] = "0,1"
    status = {}
    run_train(tmpdir, _devices=4, tenant=0, status=status)
    error_status = status[0]
    assert expected in error_status
    del os.environ["HABANA_VISIBLE_MODULES"]


@pytest.mark.parametrize(
    ("num_tenants", "cards_per_tenant"),
    [(2, 4), (4, 2)],
    ids=[
        "num_tenants_2_cards_per_tenant_4",
        "num_tenants_4_cards_per_tenant_2",
    ],
)
# Though using partial Gaudi is possible, only 2 and 4 card scenarios are recommended and supported:
# https://docs.habana.ai/en/latest/PyTorch/PT_Multiple_Tenants_on_HPU/Multiple_Workloads_Single_Docker.html#number-of-supported-gaudis-for-multi-tenancy-workload
def test_multi_tenancy_valid_cards_tenants(tmpdir, num_tenants, cards_per_tenant):
    status = spawn_tenants(tmpdir, num_tenants, cards_per_tenant)
    for _, error in status.items():
        assert error is None
