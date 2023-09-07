# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
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

import operator

import torch
from lightning_utilities.core.imports import RequirementCache, compare_version
from packaging.version import Version

from lightning_habana.utils.resources import _HABANA_FRAMEWORK_AVAILABLE, get_hpu_synapse_version  # noqa:

_HPU_SYNAPSE_GREATER_EQUAL_1_11_0 = Version(get_hpu_synapse_version()) >= Version("1.11.0")
_TORCH_LESSER_EQUAL_1_13_1 = compare_version("torch", operator.le, "1.13.1")
_TORCH_GREATER_EQUAL_2_0_0 = compare_version("torch", operator.ge, "2.0.0")
_LIGHTNING_GREATER_EQUAL_2_0_0 = compare_version("lightning", operator.ge, "2.0.0") or compare_version(
    "pytorch_lightning", operator.ge, "2.0.0"
)
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")
_KINETO_AVAILABLE = torch.profiler.kineto_available()
