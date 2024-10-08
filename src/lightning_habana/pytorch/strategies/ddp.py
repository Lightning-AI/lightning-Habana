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

if module_available("lightning"):
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch import Trainer
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning.pytorch.strategies.ddp import DDPStrategy
    from lightning.pytorch.trainer.states import TrainerFn
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.trainer.states import TrainerFn
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")
from torch import Tensor

from lightning_habana.pytorch.strategies.parallel import HPUParallelStrategy
from lightning_habana.utils.imports import _HABANA_FRAMEWORK_AVAILABLE

if _HABANA_FRAMEWORK_AVAILABLE:
    pass

log = logging.getLogger(__name__)


class HPUDDPStrategy(DDPStrategy, HPUParallelStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    strategy_name = "hpu_ddp"

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

    def setup(self, trainer: "Trainer") -> None:
        if (
            trainer.state.fn in (TrainerFn.PREDICTING, TrainerFn.TESTING)
            and trainer.precision_plugin.precision == "fp8"
        ):
            raise NotImplementedError("FP8 inference is not supported with HPUDDPStrategy yet !!!")
        return super().setup(trainer)

    def determine_ddp_device_ids(self) -> None:
        return None

    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        # Skipping DDPStrategy (first in mro) and inheriting from HPUParallelStrategy
        return HPUParallelStrategy.reduce(self, tensor, group, reduce_op)

    def _get_process_group_backend(self) -> str:
        return HPUParallelStrategy._get_process_group_backend(self)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
