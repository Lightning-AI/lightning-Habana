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


from typing import Any, Callable, Mapping, Optional, Union

import torch
from lightning_utilities import module_available
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.optim import LBFGS, Optimizer

if module_available("lightning"):
    from lightning.fabric.utilities.types import Steppable
    from lightning.pytorch import LightningModule
    from lightning.pytorch.utilities import GradClipAlgorithmType
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
    from lightning.pytorch.utilities.model_helpers import is_overridden
    from lightning.pytorch.utilities.rank_zero import WarningCache
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.types import Steppable
    from pytorch_lightning import LightningModule
    from pytorch_lightning.utilities import GradClipAlgorithmType
    from pytorch_lightning.utilities.exceptions import MisconfigurationException
    from pytorch_lightning.utilities.model_helpers import is_overridden
    from pytorch_lightning.utilities.rank_zero import WarningCache
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_habana.pytorch.plugins.precision import _PRECISION_INPUT, HPUPrecisionPlugin
from lightning_habana.utils.imports import _HPU_SYNAPSE_GREATER_EQUAL_1_14_0
from lightning_habana.utils.resources import _HABANA_FRAMEWORK_AVAILABLE

if _HPU_SYNAPSE_GREATER_EQUAL_1_14_0 and _HABANA_FRAMEWORK_AVAILABLE:
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling

_HPU_DEEPSPEED_AVAILABLE = (
    # HPU deep speed is supported only through this pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.16.0
    RequirementCache("deepspeed==0.14.0+hpu.synapse.v1.16.0")
)
if _HPU_DEEPSPEED_AVAILABLE:
    import deepspeed


warning_cache = WarningCache()


class HPUDeepSpeedPrecisionPlugin(HPUPrecisionPlugin):
    """Plugin that enables mixed precision support on HPUs.

    Args:
        precision (_PRECISION_INPUT, optional): Precision input. Defaults to "32-true".

    Raises:
        OSError: Unsupported Synapse version.
        ValueError: Invalid precision value.
        NotImplementedError: fp8 / fp16 not available.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT = "32-true",
        device: str = "hpu",
    ) -> None:
        if not _HPU_DEEPSPEED_AVAILABLE:
            raise MisconfigurationException(
                "To use the `HPUDeepSpeedPrecisionPlugin`, you must have hpu DeepSpeed installed."
                " Install it by running `pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.16.0`."
            )
        super().__init__(device=device, precision=precision)

    def backward(
        self,
        tensor: Tensor,
        model: "LightningModule",
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs back-propagation using DeepSpeed's engine.

        Args:
            tensor: the loss tensor
            model: the model to be optimized
            optimizer: ignored for DeepSpeed
            *args: additional positional arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call
            **kwargs: additional keyword arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call

        """
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        deepspeed_engine: "deepspeed.DeepSpeedEngine" = model.trainer.model
        deepspeed_engine.backward(tensor, *args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Steppable,
        model: "LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException("DeepSpeed and the LBFGS optimizer are not compatible.")
        closure_result = closure()
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `DeepSpeed`"
            )
        # DeepSpeed handles the optimizer step internally
        deepspeed_engine: "deepspeed.DeepSpeedEngine" = model.trainer.model
        return deepspeed_engine.step(**kwargs)

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """DeepSpeed handles gradient clipping internally."""
        pass

    def _enable_fp8_inference(
        self,
        module: torch.nn.Module,
        quant: bool = True,
        fp8_data_path: Optional[str] = None,
        ds_inference_kwargs: Optional[dict] = None,
    ) -> None:
        """Convert modules for fp8 inference.

        This module cannot be used with trainer.fit.

        """
        ds_inference_kwargs = {} if ds_inference_kwargs is None else ds_inference_kwargs
        if "dtype" not in ds_inference_kwargs:
            ds_inference_kwargs["dtype"] = torch.bfloat16
        assert ds_inference_kwargs["dtype"] in (torch.bfloat16, torch.float)

        htcore.quantization.hpu_set_inference_env()
        module = module.to("hpu")

        module = deepspeed.init_inference(module, **ds_inference_kwargs)
        super()._setup_fp8_inference_modules(module, quant, fp8_data_path)

    def convert_modules(
        self,
        module: torch.nn.Module,
        inference: bool = False,
        replace_layers: bool = False,
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        quant: bool = True,
        fp8_data_path: Optional[str] = None,
        ds_inference_kwargs: Optional[dict] = None,
    ) -> torch.nn.Module:
        """Enable support for fp8."""
        if inference and self.fp8_inference_available:
            self._enable_fp8_inference(module, quant, fp8_data_path, ds_inference_kwargs)
        if not inference and self.fp8_train_available:
            self._enable_fp8_training(module, replace_layers, recipe)
        return module
