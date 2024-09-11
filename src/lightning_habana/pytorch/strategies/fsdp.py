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
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Literal, Optional, Set, Type, Union

import torch
from lightning_utilities import module_available
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

if module_available("lightning"):
    import lightning.pytorch as pl
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.fabric.strategies import _StrategyRegistry
    from lightning.fabric.strategies.fsdp import (
        _move_torchmetrics_to_device,
        _setup_activation_checkpointing,
    )
    from lightning.fabric.utilities.init import _has_meta_device_parameters_or_buffers
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch.plugins.precision import Precision
    from lightning.pytorch.strategies.fsdp import FSDPStrategy
    from lightning.pytorch.utilities.rank_zero import rank_zero_warn
elif module_available("pytorch_lightning"):
    import pytorch_lightning as pl
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_fabric.strategies import _StrategyRegistry
    from lightning_fabric.strategies.fsdp import (
        _move_torchmetrics_to_device,
        _setup_activation_checkpointing,
    )
    from lightning_fabric.utilities.init import _has_meta_device_parameters_or_buffers
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning.plugins.precision import Precision
    from pytorch_lightning.strategies.fsdp import FSDPStrategy
    from pytorch_lightning.utilities.rank_zero import rank_zero_warn
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_habana.pytorch.plugins.fsdp_precision import HPUFSDPPrecision
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.pytorch.strategies.parallel import HPUParallelStrategy
from lightning_habana.utils.imports import _LIGHTNING_GREATER_EQUAL_2_3_0

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    _POLICY = Union[Set[Type[Module]], Callable[[Module, bool, int], bool], ModuleWrapPolicy]

    _SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]


log = logging.getLogger(__name__)


class HPUFSDPStrategy(FSDPStrategy, HPUParallelStrategy):
    r"""Strategy for Fully Sharded Data Parallel provided by torch.distributed on HPU.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    strategy_name = "hpu_fsdp"
    _registered_strategies: List[str] = []

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = HPUCheckpointIO(),
        precision_plugin: Optional[Precision] = HPUFSDPPrecision("bf16-mixed"),
        process_group_backend: Optional[str] = "hccl",
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        mixed_precision: Optional["MixedPrecision"] = None,
        auto_wrap_policy: Optional["_POLICY"] = None,
        activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,
        sharding_strategy: "_SHARDING_STRATEGY" = "FULL_SHARD",
        state_dict_type: Literal["full", "sharded"] = "full",
        **kwargs: Any,
    ) -> None:
        if not _LIGHTNING_GREATER_EQUAL_2_3_0:
            raise OSError("HPUFSDPStrategy requires `lightning>=2.3.0 or pytorch-lightning >= 2.3.0`.")

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            timeout=timeout,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_policy=activation_checkpointing_policy,
            sharding_strategy=sharding_strategy,
            state_dict_type=state_dict_type,
            **kwargs,
        )

    @property
    def mixed_precision_config(self) -> Optional["MixedPrecision"]:
        if self.mixed_precision:
            return self.mixed_precision
        plugin = self.precision_plugin
        if isinstance(plugin, HPUFSDPPrecision):
            return plugin.mixed_precision_config
        return None

    @property
    @override
    def precision_plugin(self) -> HPUFSDPPrecision:
        plugin = self._precision_plugin
        if plugin is not None:
            return plugin
        return HPUFSDPPrecision("bf16-mixed")

    @precision_plugin.setter
    @override
    def precision_plugin(self, precision_plugin: Optional[HPUFSDPPrecision]) -> None:
        if precision_plugin is not None and not isinstance(precision_plugin, HPUFSDPPrecision):
            raise TypeError(
                f"The FSDP strategy can only work with the `HPUFSDPPrecision` plugin, found {precision_plugin}"
            )
        self._precision_plugin = precision_plugin

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return torch.device("hpu", torch.hpu.current_device())

    def _setup_model(self, model: Module) -> Module:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in model.modules()):
            if _has_meta_device_parameters_or_buffers(model):
                rank_zero_warn(
                    "The model is already wrapped in `FSDP` but there are still parameters on the meta device."
                )
            if "auto_wrap_policy" in self.kwargs:
                # The user has wrapped their submodules manually, don't apply the auto wrap policy.
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self.kwargs["auto_wrap_policy"]
        else:
            model = FullyShardedDataParallel(
                module=model,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device,  # Index based device selection is not supported on HPU
                **self.kwargs,
            )

        _move_torchmetrics_to_device(model, self.root_device)

        # activation checkpointing needs to be set up after wrapping the model
        _setup_activation_checkpointing(model, self._activation_checkpointing_kwargs)

        return model

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    @contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            cpu_offload=self.cpu_offload,
            mixed_precision=self.mixed_precision_config,
            sharding_strategy=self.sharding_strategy,
            device_id=self.root_device,  # Index based device selection is not supported on HPU
            **self.kwargs,
        ):
            yield

    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        # Skipping FSDPStrategy (first in mro) and inheriting from HPUParallelStrategy.
        return HPUParallelStrategy.reduce(self, tensor, group, reduce_op)

    def _get_process_group_backend(self) -> str:
        return HPUParallelStrategy._get_process_group_backend(self)

    @classmethod
    def get_registered_strategies(cls) -> List[str]:
        return cls._registered_strategies

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not torch.distributed.is_available():
            return
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description="Fully Sharded Data Parallel (FSDP) training",
        )
        cls._registered_strategies.append(cls.strategy_name)
