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
import importlib.resources
import os
from contextlib import contextmanager
from typing import Any, ContextManager, Generator, Literal, Mapping, Optional, Union

import torch
from lightning_utilities import module_available
from typing_extensions import get_args

from lightning_habana.utils.imports import _HPU_SYNAPSE_GREATER_EQUAL_1_17_0
from lightning_habana.utils.resources import (
    _HABANA_FRAMEWORK_AVAILABLE,
    _INTEL_NEURAL_COMPRESSOR_AVAILABLE,
    is_fp8_available,
    is_fp16_available,
    modify_fp8_json,
)

if module_available("lightning"):
    from lightning.fabric.utilities.rank_zero import rank_zero_info
    from lightning.pytorch.plugins.precision import Precision
elif module_available("pytorch_lightning"):
    from pytorch_lightning.plugins.precision import Precision
    from pytorch_lightning.utilities.rank_zero import rank_zero_info
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

_PRECISION_INPUT = Literal["32", "32-true", "bf16", "bf16-mixed", "fp8", "16-mixed"]

_AMP_DICT = {
    "32": torch.float32,
    "32-true": torch.float32,
    "bf16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
}

if _HABANA_FRAMEWORK_AVAILABLE:
    # Required for training in fp8 using habana transformer engine
    import habana_frameworks.torch.hpex.experimental.transformer_engine as tengine
    from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling

    if _INTEL_NEURAL_COMPRESSOR_AVAILABLE:
        # Required for inference in fp8 using habana quantization toolkit
        import habana_frameworks.torch.core as htcore

        # Default quantization jsons
        MAXABS_MEASURE = str(
            importlib.resources.path("lightning_habana.pytorch.plugins.quant_config.fp8", "maxabs_measure.json")
        )
        MAXABS_QUANT = str(
            importlib.resources.path("lightning_habana.pytorch.plugins.quant_config.fp8", "maxabs_quant.json")
        )


class HPUPrecisionPlugin(Precision):
    """Plugin that enables mixed precision support on HPUs.

    Args:
        precision (_PRECISION_INPUT, optional): Precision input. Defaults to "32-true".

    Raises:
        ValueError: Invalid precision value(s).
        NotImplementedError: fp8 / fp16 not available.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT = "32-true",
        device: str = "hpu",
    ) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(accelerator='hpu', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.device = device
        self.precision = precision

        self.recipe: Union[Mapping[str, Any], "DelayedScaling"] = None
        self.replace_layers = False
        self.fp8_training_available = False
        self.fp8_inference_available = False

        if self.precision == "16-mixed":
            fp16_available, reason_no_fp16 = is_fp16_available()
            if not fp16_available:
                raise NotImplementedError(f"fp16 not supported: {reason_no_fp16}.")

        if self.precision == "fp8":
            fp8_available, reason_no_fp8 = is_fp8_available()
            if not fp8_available:
                raise NotImplementedError(f"fp8 not supported: {reason_no_fp8}.")
            self.fp8_training_available = fp8_available
            self.fp8_inference_available = (
                fp8_available and _INTEL_NEURAL_COMPRESSOR_AVAILABLE and _HPU_SYNAPSE_GREATER_EQUAL_1_17_0
            )

            rank_zero_info(
                f"fp8 training available: {self.fp8_training_available}"
                f"fp8 inference available: {self.fp8_inference_available}."
            )
            if not _INTEL_NEURAL_COMPRESSOR_AVAILABLE or not _HPU_SYNAPSE_GREATER_EQUAL_1_17_0:
                rank_zero_info(
                    "FP8 inference not available."
                    f"Synapse version found: {_HPU_SYNAPSE_GREATER_EQUAL_1_17_0}. Should be >= 1.17.09."
                    f"Intel Neural Compressor available: {_INTEL_NEURAL_COMPRESSOR_AVAILABLE}. Should be True."
                )

    def _setup_fp8_inference_config(
        self, quant: Optional[Union[bool, str, dict]] = True, fp8_data_path: Optional[str] = None
    ) -> Any:
        """Setup fp8 inference config."""
        from neural_compressor.torch.quantization import FP8Config

        fp8_config = MAXABS_QUANT if quant is True else MAXABS_MEASURE if quant is False else quant
        fp8_data_path = fp8_data_path if fp8_data_path is not None else os.environ.get("HABANA_LOGS")

        if isinstance(fp8_config, str):
            if os.path.isfile(fp8_config):
                modify_fp8_json(
                    file_path=fp8_config,
                    patch={"dump_stats_path": os.path.join(fp8_data_path, "inc_output", "measure")},
                )
                return FP8Config.from_json_file(fp8_config)
            raise FileNotFoundError

        if isinstance(fp8_config, dict):
            fp8_config["dump_stats_path"] = os.path.join(fp8_data_path, "inc_output", "measure")
            return FP8Config.from_dict(fp8_config)

        raise TypeError(f"`quant` must be either a bool, file path or a dictionary. Got {type(quant)}")

    def _enable_fp8_inference(
        self,
        module: torch.nn.Module,
        quant: Optional[Union[bool, str, dict]] = True,
        fp8_data_path: Optional[str] = None,
    ) -> None:
        """Convert module for fp8 inference.

        This module cannot be used to run trainer.fit.

        """
        htcore.hpu_set_env()
        self._setup_fp8_inference_modules(module, quant, fp8_data_path)

    def _setup_fp8_inference_modules(
        self, module: torch.nn.Module, quant: bool = True, fp8_data_path: Optional[str] = None
    ) -> None:
        """Convert module for fp8 inference."""
        from neural_compressor.torch.quantization import FP8Config, convert, prepare

        config: FP8Config = self._setup_fp8_inference_config(quant, fp8_data_path)
        module = prepare(module, config) if config.measure else convert(module, config)
        htcore.hpu_initialize(module, mark_only_scales_as_const=True)

    def _enable_fp8_training(
        self,
        module: torch.nn.Module,
        replace_layers: bool = False,
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
    ) -> None:
        """Convert module for fp8 training."""
        self.recipe = recipe
        if replace_layers:
            # In case model already contains a transformer engine modules,
            # assume user responsibility for conversion of required layers.
            if any(
                "habana_frameworks.torch.hpex.experimental.transformer_engine" in m.__module__ for m in module.modules()
            ):
                rank_zero_info(
                    f"Module {module} already contains transformer engine equivalent modules. Skipping conversion"
                )
            else:
                _replace_layers(module)

    def convert_modules(
        self,
        module: torch.nn.Module,
        inference: bool = False,
        replace_layers: bool = False,
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        quant: Optional[Union[bool, str, dict]] = True,
        fp8_data_path: Optional[str] = None,
    ) -> torch.nn.Module:
        """Convert modules for fp8 precision.

        Args:
            module (torch.nn.Module): module to convert
            inference (bool, optional): prepare modules for inference (True) / training (False). Defaults to False.
            replace_layers (bool, optional): Replace layers with transformer engine equivalent layers for fp8 training.
                Defaults to False.
            recipe (Optional[Union[Mapping[str, Any], DelayedScaling]], optional): Recipe for fp8 training.
                Defaults to None.
            quant (bool, str, dict, optional): Run fp8 inference in measurement (False) or Quant (True) mode.
                Can be used to pass a user defined dictionary or fp8 config json. Defaults to True.
            fp8_data_path (Optional[str], optional): path to dump fp8 inference data in measurement mode.
                Defaults to None.

        Returns:
            torch.nn.Module: fp8 enabled module

        """
        assert self.precision == "fp8", "HPUPrecisionPlugin.convert_modules() should only be used with precision=`fp8`."
        if inference:
            if not _HPU_SYNAPSE_GREATER_EQUAL_1_17_0:
                raise OSError("FP8 inference on HPU requires SynapsesAI >= 1.17.0")
            if not _INTEL_NEURAL_COMPRESSOR_AVAILABLE:
                raise ModuleNotFoundError(
                    "Intel neural compressor not found. " "Install it using `pip install neural-compressor`"
                )
            if self.fp8_inference_available:
                self._enable_fp8_inference(module, quant, fp8_data_path)

        if not inference and self.fp8_training_available:
            self._enable_fp8_training(module, replace_layers, recipe)
        return module

    def autocast_context_manager(self) -> Union[ContextManager[Any], torch.autocast]:
        """Return Autocast context manager."""
        if self.fp8_training_available:
            return tengine.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        return torch.autocast(device_type="hpu", dtype=_AMP_DICT[self.precision], enabled=True)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield


def _replace_layers(module: torch.nn.Module) -> None:
    """Replace layers with Transformer engine equivalent layers.

    Args: torch.nn.Module.
    Return: transformer engine equivalent of torch.nn.Module.
    List of supported modules: https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html

    Eg. torch.nn.Linear -> transformer_engine.Linear

    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            has_bias = child.bias is not None
            replacement = tengine.Linear(child.in_features, child.out_features, bias=has_bias)
            rank_zero_info(f"Replacing layer {name} with transformer engine equivalent")
            module.__setattr__(name, replacement)
        else:
            _replace_layers(child)
