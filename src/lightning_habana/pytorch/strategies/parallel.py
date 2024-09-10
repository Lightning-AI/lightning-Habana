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
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed
from lightning_utilities import module_available
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

if module_available("lightning"):
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.fabric.utilities.distributed import (
        _distributed_is_initialized,
        _init_dist_connection,
    )
    from lightning.fabric.utilities.distributed import group as _group
    from lightning.fabric.utilities.seed import reset_seed
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch import LightningModule
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
    from lightning.pytorch.plugins.precision import Precision
    from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
    from lightning.pytorch.strategies.parallel import ParallelStrategy
    from lightning.pytorch.strategies.strategy import TBroadcast
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
    from lightning.pytorch.utilities.types import STEP_OUTPUT
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_fabric.utilities.distributed import (
        _distributed_is_initialized,
        _init_dist_connection,
    )
    from lightning_fabric.utilities.distributed import group as _group
    from lightning_fabric.utilities.seed import reset_seed
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning import LightningModule
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
    from pytorch_lightning.plugins.precision import Precision
    from pytorch_lightning.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
    from pytorch_lightning.strategies.parallel import ParallelStrategy
    from pytorch_lightning.strategies.strategy import TBroadcast
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    from pytorch_lightning.utilities.types import STEP_OUTPUT
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.utils.hpu_distributed import _sync_hpu_processes_if_available
from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE

if _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl as hpu_dist

log = logging.getLogger(__name__)


class HPUParallelStrategy(ParallelStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    strategy_name = "hpu_parallel"

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self._process_group_backend = "hccl"
        self._timeout = default_pg_timeout
        self._num_nodes = 1
        self._start_method = "spawn" if self.__class__.__name__ == "HPUParallelStrategy" else None

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
        self.setup_hccl_env()
        super().setup_environment()
        if self.__class__.__name__ == "HPUParallelStrategy":
            # Strategies derived from this class should handle their own distributed setups.
            self.setup_distributed()

    def setup_hccl_env(self) -> None:
        """Initializes the HCCL environment for distributed training on HPU devices."""
        assert self._get_process_group_backend() == "hccl"
        _ws = self.cluster_environment.world_size()
        _grank = self.cluster_environment.global_rank()
        _lrank = self.cluster_environment.local_rank()
        hpu_dist.initialize_distributed_hpu(world_size=_ws, rank=_grank, local_rank=_lrank)

    def setup_distributed(self) -> None:
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        assert self.root_device.type == "hpu"
        return "hccl"

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @num_processes.setter
    def num_processes(self, num_processes: int) -> None:
        self._num_processes = num_processes

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        self._start_method = "spawn" if self._start_method is None else self._start_method
        if self._start_method == "popen":
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
        else:
            self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[0]

    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_is_initialized():
            return
        torch.distributed.barrier()

    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        if isinstance(tensor, Tensor):
            if tensor.device != self.root_device:
                tensor = tensor.to(self.root_device)
            return _sync_hpu_processes_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        output = super().training_step(*args, **kwargs)
        if self.__class__.__name__ == "HPUParallelStrategy":
            return self.reduce_loss_if_parallel(output)
        htcore.mark_step()
        return output

    def reduce_loss_if_parallel(
        self, output: Union[Tensor, dict], reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Union[Tensor, dict]:
        if isinstance(output, dict) and "loss" in output:
            output["loss"] = self.reduce(output["loss"], reduce_op=reduce_op)
        elif isinstance(output, Tensor):
            output = self.reduce(output, reduce_op=reduce_op)
        return output

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

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().predict_step(*args, **kwargs)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
