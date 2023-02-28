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
import signal
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import List

import pytest
import torch.distributed

import lightning.fabric
import lightning.pytorch
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.fabric.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_1_12
from lightning.pytorch.trainer.connectors.signal_connector import SignalConnector
from tests_pytorch import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


def pytest_addoption(parser):
    parser.addoption("--hpus", action="store", type=int, default=1, help="Number of hpus 1-8")
    parser.addoption(
        "--hmp-bf16", action="store", type=str, default="./ops_bf16_mnist.txt", help="bf16 ops list file in hmp O1 mode"
    )
    parser.addoption(
        "--hmp-fp32", action="store", type=str, default="./ops_fp32_mnist.txt", help="fp32 ops list file in hmp O1 mode"
    )


@pytest.fixture
def hpus(request):
    hpus = request.config.getoption("--hpus")
    return hpus
