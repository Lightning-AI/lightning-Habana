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
import re
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from lightning_utilities import module_available
from lightning_utilities.core.imports import package_available
from lightning_utilities.core.rank_zero import rank_zero_debug, rank_zero_warn

if module_available("lightning"):
    from lightning.fabric.utilities.exceptions import MisconfigurationException
    from lightning.fabric.utilities.types import _DEVICE
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.exceptions import MisconfigurationException
    from lightning_fabric.utilities.types import _DEVICE

_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.hpu as torch_hpu


def _parse_hpus(devices: Optional[Union[int, str, List[int]]]) -> Optional[int]:
    """Parse the HPUs given in the format as accepted by the ``Trainer`` for the ``devices`` flag.

    Args:
        devices: An integer that indicates the number of Gaudi devices to be used

    Returns:
        Either an integer or ``None`` if no devices were requested

    Raises:
        MisconfigurationException:
            If devices aren't of type `int` or `str`
    """
    if devices is not None and not isinstance(devices, (int, str)):
        raise MisconfigurationException("`devices` for `HPUAccelerator` must be int, string or None.")

    return int(devices) if isinstance(devices, str) else devices


def _parse_hpu_synapse_versions(line: str) -> Tuple[str, str]:
    """Parse the CMD output with version capture.

    Args:
        line: output of `hl-smi -v`

    Returns:
        versions of SW and fimware as string

    >>> _parse_hpu_synapse_versions("Habanalabs hl-smi/hlml version hl-1.11.0-fw-45.1.1.1 (Aug 04 2023 - 02:48:21)")
    ('1.11.0', '45.1.1.1')
    >>> _parse_hpu_synapse_versions("any string as fake CMD output")
    ('', '')
    """
    try:
        # Item "None" of "Optional[Match[str]]" has no attribute "group"
        hl = re.search(r"hl-([\d\.]+)", line).group(1)  # type: ignore[union-attr]
        fw = re.search(r"fw-([\d\.]+)", line).group(1)  # type: ignore[union-attr]
    except AttributeError:
        rank_zero_warn("Provided string does not include Habana version; check if HPU is available with `hl-smi`.")
        return "", ""
    return hl, fw


@lru_cache
def get_hpu_synapse_version() -> str:
    """Get synapse AI version."""
    try:
        proc = subprocess.Popen(["hl-smi", "-v"], stdout=subprocess.PIPE)
    # TODO: FileNotFoundError: No such file or directory: 'hl-smi'
    except (FileNotFoundError, NotADirectoryError):
        return "0.0.0"
    out = proc.communicate()[0]
    hl, fw = _parse_hpu_synapse_versions(out.decode("utf-8"))
    return hl or "0.0.0"


def get_device_stats(device: _DEVICE) -> Dict[str, Any]:
    """Return a map of the following metrics with their values.

    Include:

        - Limit: amount of total memory on HPU device.
        - InUse: amount of allocated memory at any instance.
        - MaxInUse: amount of total active memory allocated.
        - NumAllocs: number of allocations.
        - NumFrees: number of freed chunks.
        - ActiveAllocs: number of active allocations.
        - MaxAllocSize: maximum allocated size.
        - TotalSystemAllocs: total number of system allocations.
        - TotalSystemFrees: total number of system frees.
        - TotalActiveAllocs: total number of active allocations.
    """
    try:
        return torch_hpu.hpu.memory_stats(device)
    except (AttributeError, NameError):
        rank_zero_debug("HPU `get_device_stats` failed")
        return {}


@lru_cache
def device_count() -> int:
    """Return the number of HPU devices when the devices is set to auto."""
    try:
        return torch_hpu.device_count()
    except (AttributeError, NameError):
        rank_zero_debug("Function `device_count` failed, returning default count of 8.")
        return 8
