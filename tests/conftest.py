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
from pathlib import Path

import pytest
from lightning_habana import HPUAccelerator

from tests import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


def pytest_addoption(parser):
    parser.addoption("--hpus", action="store", type=int, default=1, help="Number of hpus 1-8")


@pytest.fixture()
def arg_hpus(request):
    return request.config.getoption("--hpus")


@pytest.fixture()
def device_count(pytestconfig):
    arg_hpus = int(pytestconfig.getoption("hpus"))
    if not arg_hpus:
        assert HPUAccelerator.auto_device_count() >= 1
        return 1
    assert arg_hpus <= HPUAccelerator.auto_device_count(), "More hpu devices asked than present"
    return arg_hpus


@pytest.fixture()
def _check_distributed(device_count):
    if device_count <= 1:
        pytest.skip("Distributed test does not run on single HPU")
