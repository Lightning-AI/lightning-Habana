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

import torch
from lightning_utilities.core.imports import module_available
from torch import Tensor

if module_available("lightning"):
    from lightning.fabric.utilities.distributed import _sync_ddp
    from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
    from lightning.fabric.utilities.types import ReduceOp
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.distributed import _sync_ddp
    from lightning_fabric.utilities.rank_zero import rank_zero_warn, rank_zero_info
    from lightning_fabric.utilities.types import ReduceOp

from typing import Any, Optional, Union

# Supported ReduceOps: https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/C_API.html#hcclredop-t
supported_reduce_ops = {
    "sum": ReduceOp.SUM,
    "min": ReduceOp.MIN,
    "max": ReduceOp.MAX,
}


def _distributed_available() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_backend() == "hccl"
    )


def _is_reduce_op_supported(reduce_op: Union[ReduceOp, str]) -> bool:
    """Function to check if reduce_op is supported with hccl backend."""
    reduce_op = reduce_op.lower() if isinstance(reduce_op, str) else reduce_op
    if reduce_op in ("mean", "avg") or reduce_op == ReduceOp.AVG:
        rank_zero_warn(f"{reduce_op} is not supported with HCCL. Going to simulate it")
        return True
    if reduce_op not in supported_reduce_ops and not any(reduce_op is op for op in supported_reduce_ops.values()):
        raise TypeError(
            f"Unsupported ReduceOp {reduce_op}. Supported ops in HCCL are: {', '.join(supported_reduce_ops)}"
        )
    return True


def _sync_ddp_if_available(
    result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
) -> Tensor:
    """Function to reduce a tensor across worker processes during distributed training.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.

    Return:
        reduced value

    """
    if _distributed_available() and _is_reduce_op_supported(reduce_op):
        return _sync_ddp_hpu(result, group=group, reduce_op=reduce_op)
    return result


def _sync_ddp_hpu(
    result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
) -> Tensor:
    """Reduces a tensor across several distributed processes.

    This operation is performed in-place, meaning the result will be placed back into the input tensor on all processes.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.

    Return:
        The reduced value.

    """
    # Simulate mean using sum
    reduce_op = reduce_op.lower() if isinstance(reduce_op, str) else reduce_op
    op = ReduceOp.SUM if (reduce_op == ReduceOp.AVG or reduce_op in ("mean", "avg")) else reduce_op
    result = _sync_ddp(result, group, op)

    if reduce_op == ReduceOp.AVG or reduce_op in ("mean", "avg"):
        # Compute mean from sum
        group = torch.distributed.group.WORLD if group is None else group
        world_size = torch.distributed.get_world_size(group)

        # HPU doesn't support Long types, forcefully set it to float
        if result.type() in (
            "torch.LongTensor",
            "torch.hpu.LongTensor",
        ):
            rank_zero_info("Long tensor unsupported on HPU, casting to float")
            result = result.float()
        return result.div_(world_size)
    return result
