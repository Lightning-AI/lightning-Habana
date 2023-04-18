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

import time
from typing import Tuple, Any

import numpy as np
import torch.utils.data

import lightning_habana.pytorch.datamodule.utils as utils
from lightning_habana.utils.imports import _HPU_AVAILABLE

TRAIN_RESIZE_DIM = 224
EVAL_RESIZE_DIM = 256
CROP_DIM = 224

DECODER_SCALE_MIN = 0.08
DECODER_SCALE_MAX = 1.0
DECODER_RATIO_MIN = 0.75
DECODER_RATIO_MAX = 1.3333333333333333

USE_HORIZONTAL_FLIP = 1
FLIP_PROBABILITY = 0.5

RGB_MEAN_VALUES = [0.485, 0.456, 0.406]
RGB_STD_VALUES = [0.229, 0.224, 0.225]
RGB_MULTIPLIER = 255

EVAL_CROP_X = 0.5
EVAL_CROP_Y = 0.5

if _HPU_AVAILABLE:
    try:
        from habana_frameworks.mediapipe import fn
        from habana_frameworks.mediapipe.media_types import decoderStage, dtype, ftype, imgtype, randomCropType
        from habana_frameworks.mediapipe.mediapipe import MediaPipe
        from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
        from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUResnetPytorchIterator
    except ImportError:
        raise ModuleNotFoundError("habana_dataloader package is not installed.")


class ResnetMediaPipe(MediaPipe):
    """resnet classifier with media pipe.

    :params is_training: True if ResnetMediaPipe handles training data, False in case of evaluation.
    :params root: path from which to load the images.
    :params batch_size: mediapipe output batch size.
    :params shuffle: whether images have to be shuffled.
    :params drop_last: whether to drop the last incomplete batch or round up.
    :params queue_depth: Number of preloaded image batches for every slice in mediapipe. <1/2/3>
    :params num_instances: number of devices.
    :params instance_id: instance id of current device.
    :params device: media device to run mediapipe on. <hpu/hpu:0>
    """

    instance_count = 0

    def __init__(
        self,
        is_training=False,
        root=None,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        queue_depth=1,
        num_instances=1,
        instance_id=0,
        device=None,
        seed=None,
    ) -> None:
        self.is_training = is_training
        self.root = root
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_instances = num_instances
        self.instance_id = instance_id

        ResnetMediaPipe.instance_count += 1
        pipe_name = f"{self.__class__.__name__}:{ResnetMediaPipe.instance_count}"
        pipe_name = str(pipe_name)

        super().__init__(device=device, batch_size=batch_size, prefetch_depth=queue_depth, pipe_name=pipe_name)

        if seed is None:
            seed = int(time.time_ns() % (2**31 - 1))
        resize_dim = TRAIN_RESIZE_DIM if self.is_training else EVAL_RESIZE_DIM

        self.input = fn.ReadImageDatasetFromDir(
            dir=self.root,
            format="JPEG",
            seed=seed,
            shuffle=self.shuffle,
            drop_remainder=self.drop_last,
            label_dtype=dtype.UINT32,
            num_slices=self.num_instances,
            slice_index=self.instance_id,
        )

        if self.is_training is True:
            self.decode = fn.ImageDecoder(
                output_format=imgtype.RGB_P,
                resize=[resize_dim, resize_dim],
                resampling_mode=ftype.BI_LINEAR,
                random_crop_type=randomCropType.RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP,
                scale_min=DECODER_SCALE_MIN,
                scale_max=DECODER_SCALE_MAX,
                ratio_min=DECODER_RATIO_MIN,
                ratio_max=DECODER_RATIO_MAX,
                seed=seed,
                decoder_stage=decoderStage.ENABLE_ALL_STAGES,
            )
        else:
            self.decode = fn.ImageDecoder(
                output_format=imgtype.RGB_P,
                resize=[resize_dim, resize_dim],
                resampling_mode=ftype.BI_LINEAR,
                decoder_stage=decoderStage.ENABLE_ALL_STAGES,
            )

        if self.is_training is True:
            self.random_flip_input = fn.MediaFunc(
                func=RandomFlipFunction, shape=[self.getBatchSize()], dtype=dtype.UINT8, seed=seed
            )
            self.random_flip = fn.RandomFlip(horizontal=USE_HORIZONTAL_FLIP)

        normalized_mean = np.array([m * RGB_MULTIPLIER for m in RGB_MEAN_VALUES], dtype=np.float32)
        normalized_std = np.array([1 / (s * RGB_MULTIPLIER) for s in RGB_STD_VALUES], dtype=np.float32)

        # Define Constant tensors
        self.norm_mean = fn.MediaConst(data=normalized_mean, shape=[1, 1, normalized_mean.size], dtype=dtype.FLOAT32)
        self.norm_std = fn.MediaConst(data=normalized_std, shape=[1, 1, normalized_std.size], dtype=dtype.FLOAT32)

        if self.is_training is True:
            self.cmn = fn.CropMirrorNorm(crop_w=CROP_DIM, crop_h=CROP_DIM, crop_d=0, dtype=dtype.FLOAT32)
        else:
            self.cmn = fn.CropMirrorNorm(
                crop_w=CROP_DIM,
                crop_h=CROP_DIM,
                crop_d=0,
                crop_pos_x=EVAL_CROP_X,
                crop_pos_y=EVAL_CROP_Y,
                dtype=dtype.FLOAT32,
            )

    def definegraph(self) -> Tuple[Any, Any]:
        """Defines the media graph for Resnet.

        :returns : output images, labels
        """
        jpegs, data = self.input()
        images = self.decode(jpegs)

        if self.is_training is True:
            flip = self.random_flip_input()
            images = self.random_flip(images, flip)

        mean = self.norm_mean()
        std = self.norm_std()
        images = self.cmn(images, mean, std)

        return images, data


class RandomFlipFunction(media_function):
    """Randomly generate input for RandomFlip media node.

    :params params: random_flip_func specific params.
                    shape: output shape
                    dtype: output data type
                    seed: seed to be used
    """

    def __init__(self, params):
        self.np_shape = params["shape"][::-1]
        self.np_dtype = params["dtype"]
        self.seed = params["seed"]
        self.rng = np.random.default_rng(self.seed)

    def __call__(self) -> Any:
        """:returns : randomly generated binary output per image."""
        probabilities = [1.0 - FLIP_PROBABILITY, FLIP_PROBABILITY]
        random_flips = self.rng.choice([0, 1], p=probabilities, size=self.np_shape)
        random_flips = np.array(random_flips, dtype=self.np_dtype)
        return random_flips


class MediaApiDataLoader(torch.utils.data.DataLoader):
    """Helper to construct resnet media pipe dataloader."""

    def __init__(
        self,
        dataset,
        sampler,
        batch_size,
        num_workers,
        pin_memory=True,
        pin_memory_device=None,
        is_training=False,
        seed=None,
        shuffle=False,
        drop_last=False,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = None

        self.shuffle = isinstance(self.sampler, torch.utils.data.RandomSampler) or (
            isinstance(self.sampler, torch.utils.data.distributed.DistributedSampler) and (self.sampler.shuffle is True)
        )

        root = self.dataset.root
        if "train" in root:
            is_training = True
            self.shuffle = True
        else:
            self.shuffle = False

        device_string = utils.get_device_string()
        num_instances = utils.get_world_size()
        instance_id = utils.get_rank()
        queue_depth = 3

        pipeline = ResnetMediaPipe(
            is_training=is_training,
            root=root,
            batch_size=batch_size,
            shuffle=self.shuffle,
            drop_last=False,
            queue_depth=queue_depth,
            num_instances=num_instances,
            instance_id=instance_id,
            device=device_string,
            seed=seed,
        )

        self.iterator = HPUResnetPytorchIterator(mediapipe=pipeline)
        print("Running with Media API")

    def __len__(self) -> Any:
        """Return length of the iterator."""
        return len(self.iterator)

    def __iter__(self) -> Any:
        """Returns iterator."""
        return iter(self.iterator)
