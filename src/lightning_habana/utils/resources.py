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
import json
import re
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from lightning_utilities import module_available
from lightning_utilities.core.imports import package_available
from lightning_utilities.core.rank_zero import rank_zero_debug, rank_zero_warn

_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")
_INTEL_NEURAL_COMPRESSOR_AVAILABLE = package_available("neural_compressor")

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.hpu as torch_hpu

if module_available("lightning"):
    from lightning.fabric.utilities.exceptions import MisconfigurationException
    from lightning.fabric.utilities.types import _DEVICE
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.exceptions import MisconfigurationException
    from lightning_fabric.utilities.types import _DEVICE


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
        versions of SW and firmware as string

    >>> _parse_hpu_synapse_versions("Habanalabs hl-smi/hlml version hl-1.11.0-fw-45.1.1.1 (Aug 04 2023 - 02:48:21)")
    ('1.11.0', '45.1.1.1')
    >>> _parse_hpu_synapse_versions("any string as fake CMD output")
    ('', '')

    """
    hl = fw = ""
    try:
        # Item "None" of "Optional[Match[str]]" has no attribute "group"
        hl = re.search(r"hl-([\d\.]+)", line).group(1)  # type: ignore[union-attr]
        fw = re.search(r"fw-([\d\.]+)", line).group(1)  # type: ignore[union-attr]
    except AttributeError:
        rank_zero_warn("Provided string does not include Habana version; check if HPU is available with `hl-smi`.")

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


@lru_cache
def get_device_name_from_backend() -> str:
    """Return the name of the HPU device."""
    try:
        # this opens up a device to retrieve the name
        return torch_hpu.get_device_name()
    except (AttributeError, NameError):
        # return GAUDI as default name
        return "GAUDI"


def _parse_for_device_name(line: str) -> str:
    """Parse the CMD output with version capture.

    Args:
        line: output of `hl-smi -L`

    Returns:
        device name

    >>> _parse_for_device_name("Zephyr 2.7.2-hl-gaudi2-1.17.2-fw-51.5.1-sec-9")
    ('GAUDI2')
    >>> _parse_for_device_name("any other input")
    ('GAUDI')

    """
    name = "GAUDI"
    try:
        name = name + re.search(r"hl-gaudi([\d\-])", line).group(1)  # type: ignore[union-attr]
    except AttributeError:
        rank_zero_warn("Provided string does not include device name; check if HPU is available with `hl-smi -L`.")

    return name.replace("-", "")


@lru_cache
def get_device_name_from_hlsmi() -> str:
    """Get hpu device name from hl-smi."""
    try:
        proc = subprocess.Popen(["hl-smi", "-L"], stdout=subprocess.PIPE)
    except (FileNotFoundError, NotADirectoryError):
        # if hl-smi is not present, we open a device to get the name
        return get_device_name_from_backend()
    out = proc.communicate()[0]
    return _parse_for_device_name(out.decode("utf-8"))


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


@lru_cache
def is_fp8_available() -> Tuple[bool, str]:
    """Returns a bool indicating if fp8 is available."""
    if not _HABANA_FRAMEWORK_AVAILABLE:
        raise OSError("Habana Frameworks required for training on Habana devices.")

    if get_device_name_from_hlsmi() == "GAUDI":
        return False, "FP8 not supported on Gaudi, Gaudi2 or higher required"
    return True, ""


@lru_cache
def is_fp16_available() -> Tuple[bool, str]:
    """Returns a bool indicating if fp16 is available."""
    if not _HABANA_FRAMEWORK_AVAILABLE:
        raise OSError("Habana Frameworks required for training on Habana devices.")

    if get_device_name_from_hlsmi() == "GAUDI":
        return False, "FP16 not supported on Gaudi, Gaudi2 or higher required."
    return True, ""


def modify_fp8_json(file_path: str, patch: dict) -> None:
    """Edit a specific entry in a JSON file.

    Parameters:
        file_path (str): The path to the JSON file.
        patch (dict): Entries to patch in json

    Returns:
        None

    """
    # Load the JSON file
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    # Edit the specified entries
    for key, value in patch.items():
        data[key] = value

    # Update json
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)
