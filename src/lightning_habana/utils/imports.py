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
import os

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from lightning_utilities.core.imports import compare_version, package_available  # noqa: E402

_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")
if _HABANA_FRAMEWORK_AVAILABLE:
    from habana_frameworks.torch.utils.library_loader import is_habana_available

    _HPU_AVAILABLE: bool = is_habana_available()
else:
    _HPU_AVAILABLE = False

_TORCH_LESSER_EQUAL_1_13_1 = compare_version("torch", operator.le, "1.13.1")
