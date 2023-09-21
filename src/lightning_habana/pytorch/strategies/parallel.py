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
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.utilities.distributed import group as _group
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch import LightningModule
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning.pytorch.strategies.ddp import DDPStrategy
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.utilities.distributed import group as _group
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning import LightningModule
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.ddp import DDPStrategy
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.utils.hpu_distributed import _sync_ddp_if_available
from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl as hpu_dist

log = logging.getLogger(__name__)


class HPUParallelStrategy(DDPStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    strategy_name = "hpu_parallel"

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = "hccl",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            ddp_comm_state=ddp_comm_state,
            ddp_comm_hook=ddp_comm_hook,
            ddp_comm_wrapper=ddp_comm_wrapper,
            model_averaging_period=model_averaging_period,
            process_group_backend=process_group_backend,
            **kwargs,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:  # type: ignore[has-type]
            self._checkpoint_io = HPUCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = HPUCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io  # type: ignore

    def setup_environment(self) -> None:
        if self._process_group_backend == "hccl":
            # this env is used in overrides to check the backend initiated
            os.environ["HCCL_DISTRIBUTED_BACKEND"] = str(1)
            _ws = self.cluster_environment.world_size()
            _grank = self.cluster_environment.global_rank()
            _lrank = self.cluster_environment.local_rank()
            hpu_dist.initialize_distributed_hpu(world_size=_ws, rank=_grank, local_rank=_lrank)
        super().setup_environment()

    def determine_ddp_device_ids(self) -> None:
        return None

    def broadcast(self, obj: object, src: int = 0) -> object:
        if not torch.distributed.is_available():
            return obj

        obj = [obj]
        if self.global_rank != src:
            obj = [None]

        _hpu_broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def on_after_backward(self) -> None:
        # Break lazy accumulation of graph after fwd+bwd
        htcore.mark_step()

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Callable[[], Any],
        model: Optional[Union[LightningModule, Module]] = None,
        **kwargs: Any,
    ) -> Any:
        optimizer_output = super().optimizer_step(optimizer, closure, model, **kwargs)
        # Break lazy accumulation of graph after optimizer
        htcore.mark_step()
        return optimizer_output

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().predict_step(batch, batch_idx)

    def reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("HCCL_DISTRIBUTED_BACKEND", None)


# The code underneath is taken from PyTorch `torch/distributed/distributed_c10d.py`
# the distributed backend and tensor type updates for habana backend is done here before broadcast
def _hpu_broadcast_object_list(object_list, src=0, group=None, device=None):  # type: ignore
    from torch.distributed import Backend, _rank_not_in_group, broadcast, get_backend, get_rank
    from torch.distributed.distributed_c10d import _object_to_tensor, _tensor_to_object

    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(*[_object_to_tensor(obj, device) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    is_hpu_backend = os.environ.get("HCCL_DISTRIBUTED_BACKEND") == "1"
    if device is not None:
        if is_nccl_backend and device.type != "cuda":
            raise ValueError("device type must be cuda for nccl backend")
        current_device = device
    else:
        current_device = torch.device("cpu")
        if is_nccl_backend:
            # See note about using torch.cuda.current_device() here in
            # docstring. We cannot simply use my_rank since rank == device is
            # not necessarily true.
            current_device = torch.device("cuda", torch.cuda.current_device())
    if is_nccl_backend:
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    elif is_hpu_backend:
        current_device = torch.device("hpu")
        # Workaround: HPU doesn't not support long tensors for collectives
        if (object_sizes_tensor.type() == "torch.LongTensor") or (object_sizes_tensor.type() == "torch.hpu.LongTensor"):
            object_sizes_tensor = object_sizes_tensor.int()
        else:
            print("unhandled hpu object_sizes_tensor type :: ", object_sizes_tensor.type())
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(),
            dtype=torch.uint8,
        )

    if is_nccl_backend or is_hpu_backend:
        object_tensor = object_tensor.to(current_device)

    broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)
