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

from lightning_habana.utils.imports import _HPU_SYNAPSE_GREATER_EQUAL_1_11_0, _HPU_SYNAPSE_GREATER_EQUAL_1_14_0
from lightning_habana.utils.resources import (
    _HABANA_FRAMEWORK_AVAILABLE,
    _HABANA_QUANTIZATION_TOOLKIT_AVAILABLE,
    is_fp8_available,
    modify_fp8_json,
)

if module_available("lightning"):
    from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
    from lightning.pytorch.plugins.precision import Precision
elif module_available("pytorch_lightning"):
    from pytorch_lightning.plugins.precision import Precision
    from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

_PRECISION_INPUT = Literal["32", "32-true", "bf16", "bf16-mixed", "fp8"]

if _HPU_SYNAPSE_GREATER_EQUAL_1_14_0 and _HABANA_FRAMEWORK_AVAILABLE:
    # Required for training in fp8 using habana transformer engine
    import habana_frameworks.torch.hpex.experimental.transformer_engine as tengine
    from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling

    if _HABANA_QUANTIZATION_TOOLKIT_AVAILABLE:
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
        precision: to enable ``torch.bfloat16`` (``'bf16-mixed'``).
        device: The device for ``torch.autocast``.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT,
        device: str = "hpu",
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        replace_layers: bool = False,
    ) -> None:
        if not _HPU_SYNAPSE_GREATER_EQUAL_1_11_0:
            raise OSError("HPU precision plugin requires `Synapse AI release >= 1.11.0`.")
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(accelerator='hpu', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = precision
        self.replace_layers = False
        self.device = device

        if any([recipe, replace_layers]) and precision != "fp8":
            rank_zero_warn(f"Precision is not 'fp8'. Params {recipe=} and {replace_layers=} will not be set.")

        self.recipe = None
        self.fp8_train_available = False
        self.fp8_inference_available = False

        if self.precision == "fp8":
            fp8_available, reason_no_fp8 = is_fp8_available()
            if not fp8_available:
                raise NotImplementedError(f"fp8 not supported: {reason_no_fp8}.")
            self.recipe = recipe
            self.fp8_train_available = fp8_available
            self.fp8_inference_available = fp8_available and _HABANA_QUANTIZATION_TOOLKIT_AVAILABLE
            self.replace_layers = replace_layers

            rank_zero_info(
                f"fp8 training available: {self.fp8_train_available}. "
                f"fp8 inference available: {self.fp8_inference_available}."
            )

    def _setup_fp8_quant_config(self, quant: bool = True, fp8_data_path: Optional[str] = None) -> None:
        """Setup QUANT_CONFIG for before importing HQT."""
        if os.environ.get("QUANT_CONFIG", None) is None:
            # Use default jsons in case one is not provided via env variable
            fp8_data_path = fp8_data_path if fp8_data_path is not None else os.environ.get("HABANA_LOGS")
            assert fp8_data_path is not None
            # Create a copy in fp8_dump_path to avoid modifying package jsons.
            fp8_json = MAXABS_QUANT if quant else MAXABS_MEASURE
            if fp8_data_path is not None:
                modify_fp8_json(
                    file_path=fp8_json,
                    patch={
                        "dump_stats_path": os.path.join(fp8_data_path, "hqt"),
                        "dump_stats_xlsx_path": os.path.join(fp8_data_path, "hqt", "fp8stats.xlsx"),
                    },
                )
            os.environ["QUANT_CONFIG"] = fp8_json

    def _enable_fp8_inference(
        self, module: torch.nn.Module, quant: bool = True, fp8_data_path: Optional[str] = None
    ) -> None:
        """Convert module for fp8 inference.

        This module cannot be used to run trainer.fit.

        """
        htcore.hpu_set_env()
        module = module.to("hpu")
        self._setup_fp8_inference_modules(module, quant, fp8_data_path)

    def _setup_fp8_inference_modules(
        self, module: torch.nn.Module, quant: bool = True, fp8_data_path: Optional[str] = None
    ) -> None:
        """Convert module for fp8 inference."""
        try:
            self._setup_fp8_quant_config(quant, fp8_data_path)
            from quantization_toolkit import habana_quantization_toolkit

            habana_quantization_toolkit.prep_model(module)
            htcore.hpu_initialize(module)
        except FileNotFoundError as e:
            print(
                "Please run the fp8 measurement using a portion of data and try again. "
                "Use HPUPrecisionPlugin.convert_modules(module, inference=True, quant=False) "
                "and run trainer.fit() to dump measurement data."
            )
            raise e
        except ModuleNotFoundError as e:
            print("quantization_toolkit not found. Please install it using `pip install habana_quantization_toolkit`.")
            raise e

    def _enable_fp8_training(self, module: torch.nn.Module) -> None:
        """Convert module for fp8 training."""
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
        quant: bool = True,
        fp8_data_path: Optional[str] = None,
    ) -> torch.nn.Module:
        """Enable support for fp8."""
        if inference is True and self.fp8_inference_available:
            self._enable_fp8_inference(module, quant, fp8_data_path)
        if self.fp8_train_available is True and self.replace_layers is True and inference is False:
            self._enable_fp8_training(module)
        return module

    def autocast_context_manager(self) -> Union[ContextManager[Any], torch.autocast]:
        """Return Autocast context manager."""
        if self.fp8_train_available:
            return _nested_precision_cm(fp8_enabled=(self.precision == "fp8"), recipe=self.recipe)
        return torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True)

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


@contextmanager
def _nested_precision_cm(
    fp8_enabled: bool, recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]]
) -> Generator[Any, Any, Any]:
    """CM to nest fp8 precision with torch.autocast.

    This enables the ops that do not support fp8 to run with torch autocast.

    """
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True), tengine.fp8_autocast(
        enabled=fp8_enabled, fp8_recipe=recipe
    ):
        yield
