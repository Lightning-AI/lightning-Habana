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


import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
from lightning_utilities import module_available
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module

if module_available("lightning"):
    from lightning.fabric.accelerators import Accelerator
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.fabric.plugins.precision import Precision
    from lightning.fabric.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
    from lightning.fabric.strategies.parallel import ParallelStrategy
    from lightning.fabric.strategies.strategy import TBroadcast
    from lightning.fabric.utilities.distributed import (
        _distributed_is_initialized,
        _get_default_process_group_backend_for_device,
        _init_dist_connection,
    )
    from lightning.fabric.utilities.distributed import group as _group
    from lightning.fabric.utilities.rank_zero import rank_zero_only
    from lightning.fabric.utilities.seed import reset_seed
    from lightning.fabric.utilities.types import Optimizable, ReduceOp
elif module_available("pytorch_lightning"):
    from lightning_fabric.accelerators import Accelerator
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_fabric.plugins.precision import Precision
    from lightning_fabric.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
    from lightning_fabric.strategies.parallel import ParallelStrategy
    from lightning_fabric.strategies.strategy import TBroadcast
    from lightning_fabric.utilities.distributed import (
        _distributed_is_initialized,
        _get_default_process_group_backend_for_device,
        _init_dist_connection,
    )
    from lightning_fabric.utilities.distributed import group as _group
    from lightning_fabric.utilities.rank_zero import rank_zero_only
    from lightning_fabric.utilities.seed import reset_seed
    from lightning_fabric.utilities.types import Optimizable, ReduceOp
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")


from lightning_habana import HPU_AVAILABLE
from lightning_habana.fabric.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.utils.hpu_distributed import _sync_hpu_processes_if_available
from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE, _TORCH_LESSER_EQUAL_1_13_1

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
        parallel_devices: Optional[List[torch.device]] = [torch.device("hpu")] * HPUAccelerator.auto_device_count(),
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        **kwargs: Any,
    ) -> None:
        if not HPU_AVAILABLE:
            raise ValueError("`HPUParallelStrategy` requires HPU devices to run")

        self._process_group_backend: Optional[str] = "hccl"
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._process_group_backend = "hccl"
        self._timeout = default_pg_timeout
        self._num_nodes = 1
        self._start_method = "spawn" if self.strategy_name == "hpu_parallel" else None

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:  # type: ignore
            self._checkpoint_io = HPUCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    def setup_environment(self) -> None:
        super().setup_environment()
        if self.strategy_name == "hpu_parallel":
            # Strategies derived from this class should handle their own distributed setups.
            self.setup_distributed()
        self.setup_hccl_env()

    def setup_hccl_env(self):
        assert self._process_group_backend == "hccl"
        # this env is used in overrides to check the backend initiated
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
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

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
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_processes = num_processes

    def setup_module(self, module: Module) -> Module:
        """Performs setup for the model, e.g., by wrapping it by another class."""
        # Fabric doesn't support nn.Module with wrapped attributes currently.
        # Refer https://github.com/Lightning-AI/pytorch-lightning/issues/19307 for the description.
        # It is a workaround as default wrapper is overridden for HPU backend.
        if hasattr(Module, "original__get_attr__"):
            if module_available("lightning"):
                from lightning.fabric.wrappers import _FabricModule
            elif module_available("pytorch_lightning"):
                from lightning_fabric.wrappers import _FabricModule

            Module.__getattr__ = _FabricModule.original__get_attr__  # type: ignore

        return super().setup_module(module)

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        self._start_method = "spawn" if self._start_method is None else self._start_method
        if self._start_method == "popen":
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
        else:
            self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

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

    def reduce_loss_if_parallel(self, output: Union[Tensor, dict], reduce_op: Optional[Union[ReduceOp, str]] = "mean"):
        if isinstance(output, dict) and "loss" in output:
            output["loss"] = self.reduce(output["loss"], reduce_op=reduce_op)
        elif isinstance(output, Tensor):
            output = self.reduce(output, reduce_op=reduce_op)
        return output

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        output = super().training_step(*args, **kwargs)
        if self.strategy_name == "hpu_parallel":
            return self.reduce_loss_if_parallel(output)
        htcore.mark_step()
        return output

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return super().predict_step(*args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        optimizer_output = super().optimizer_step(optimizer=optimizer, **kwargs)
        if _TORCH_LESSER_EQUAL_1_13_1:
            # Break lazy accumulation of graph after optimizer
            htcore.mark_step()
        return optimizer_output

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
