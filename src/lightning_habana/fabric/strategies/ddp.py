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
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Union

import torch.distributed
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.accelerators import Accelerator
    from lightning.fabric.plugins import CheckpointIO
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
    from lightning.fabric.plugins.precision import Precision
    from lightning.fabric.strategies.ddp import DDPStrategy
    from lightning.fabric.utilities.types import ReduceOp
elif module_available("pytorch_lightning"):
    from lightning_fabric.accelerators import Accelerator
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
    from lightning_fabric.plugins.precision import Precision
    from lightning_fabric.strategies.ddp import DDPStrategy
    from lightning_fabric.utilities.types import ReduceOp
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")


from lightning_habana import HPU_AVAILABLE
from lightning_habana.fabric.accelerator import HPUAccelerator
from lightning_habana.fabric.strategies.parallel import HPUParallelStrategy

log = logging.getLogger(__name__)


class HPUDDPStrategy(DDPStrategy, HPUParallelStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    strategy_name = "hpu_ddp"

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = [torch.device("hpu")] * HPUAccelerator.auto_device_count(),
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = "hccl",
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        **kwargs: Any,
    ) -> None:
        if not HPU_AVAILABLE:
            raise ValueError("`HPUDDPStrategy` requires HPU devices to run")

        self._process_group_backend: Optional[str] = "hccl"
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method=start_method,
            **kwargs,
        )

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    def determine_ddp_device_ids(self) -> None:
        return None

    def reduce(
        self, tensor: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> torch.Tensor:
        # Skipping FSDPStrategy (first in mro) and inheriting from HPUParallelStrategy.
        return HPUParallelStrategy.reduce(self, tensor, group, reduce_op)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
