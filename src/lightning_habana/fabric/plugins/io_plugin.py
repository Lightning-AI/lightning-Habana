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

import os
from typing import Any, Dict, Optional

import torch
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.plugins import TorchCheckpointIO
    from lightning.fabric.utilities import move_data_to_device
    from lightning.fabric.utilities.cloud_io import _atomic_save, get_filesystem
    from lightning.fabric.utilities.rank_zero import rank_zero_warn
    from lightning.fabric.utilities.types import _PATH
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins import TorchCheckpointIO
    from lightning_fabric.utilities import move_data_to_device
    from lightning_fabric.utilities.cloud_io import _atomic_save, get_filesystem
    from lightning_fabric.utilities.rank_zero import rank_zero_warn
    from lightning_fabric.utilities.types import _PATH
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")


class HPUCheckpointIO(TorchCheckpointIO):
    """CheckpointIO to save checkpoints for HPU training strategies."""

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``
        Raises:
            TypeError:
                If ``storage_options`` arg is passed in.
        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = move_data_to_device(checkpoint, torch.device("cpu"))
        try:
            # write the checkpoint dictionary to the provided path
            _atomic_save(checkpoint, path)
        except AttributeError as err:
            # todo: is this try catch necessary still?
            # https://github.com/Lightning-AI/lightning/pull/431
            # TODO(fabric): Fabric doesn't support hyperparameters in the checkpoint, so this should be refactored
            key = "hyper_parameters"
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            _atomic_save(checkpoint, path)
