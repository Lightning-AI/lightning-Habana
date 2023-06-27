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

from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.hpu as torch_hpu

import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """Check distributed backend is initialized or not."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get World size."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the worker."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if this is the main process using rank."""
    return get_rank() == 0


def is_gaudi() -> bool:
    """Check if device is of Gaudi type."""
    return torch_hpu.get_device_name() == "GAUDI"


def is_gaudi2() -> bool:
    """Check if device is of Gaudi2  type."""
    return torch_hpu.get_device_name() == "GAUDI2"


def get_device_string() -> str:
    """Get the device string name."""
    if is_gaudi():
        return "gaudi"
    if is_gaudi2():
        return "gaudi2"

    raise ValueError("Unsupported device")
