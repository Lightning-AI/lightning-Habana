:orphan:

.. _hpu_intermediate:

Accelerator: HPU Training
=========================
This document offers instructions to Gaudi chip users who want to conserve memory and scale models using mixed-precision training.

----

Enable Mixed Precision
----------------------

With Lightning, you can leverage mixed precision training on HPUs. By default, HPU training
uses 32-bit precision. To enable mixed precision, set the ``precision`` flag.

.. code-block:: python

    from lightning_habana.pytorch.accelerator import HPUAccelerator

    trainer = Trainer(devices=1, accelerator=HPUAccelerator(), precision="bf16-mixed")

----

Customize Mixed Precision Using Autocast
----------------------------------------

Lightning supports following methods to enable mixed precision training with HPU.

**HPUPrecisionPlugin**

HPUPrecisionPlugin, :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin` enables mixed precision training on Habana devices.

In addition to the default settings, you can choose to override these defaults and provide your own BF16 (LOWER_LIST) and FP32 (FP32_LIST)
The `LOWER_LIST` and `FP32_LIST` environment variables must be set before any instances begin.
HPUPrecisionPlugin supports `bf16-mixed` and `16-mixed` for mixed precision training. It is advised to use `bf16-mixed` over `16-mixed` where possible.

The following is an excerpt from an MNIST example implemented on a single HPU.

.. code-block:: python

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin

    # Initialize a trainer with HPU accelerator for HPU strategy for single device,
    # with HPU precision plugin for autocast
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=1,
        plugins=[
            HPUPrecisionPlugin(
                precision="bf16-mixed",
            )
        ],
    )

    # Init our model
    model = LitClassifier()
    # Init the data
    dm = MNISTDataModule(batch_size=batch_size)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)



**Native PyTorch torch.autocast()**

For more granular control over with mixed precision training, one can use torch.autocast from native PyTorch.

Instances of autocast serve as context managers or decorators that allow regions of your script to run in mixed precision.


.. code-block:: python

    import torch
    from lightning import Trainer

    class AutocastModelCM(nn.Module):
        # Autocast can be used as a context manager to the required code block.
        def forward(self, input):
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            ...
            return

    class AutocastModelDecorator(nn.Module):
        # Autocast can be used as a decorator to the required code block.
        @torch.autocast(device_type="hpu", dtype=torch.bfloat16)
        def forward(self, input):
            ...
            return

    # Initialize a trainer with HPU accelerator for HPU strategy for single device,
    # with mixed precision using overridden HMP settings
    trainer = Trainer(
        accelerator="hpu",
        devices=1,
    )

    # Init our model
    model = AutocastModelCM()
    # Init the data
    dm = MNISTDataModule(batch_size=batch_size)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)


`torch.autocast` context manager allows fine-tuning of mixed precision training with `enabled` parameter.
It can be used alongside `HPUPrecisionPlugin`, which globally enables mixed precision, while local `torch.autocast` contexts can disable it for particular model parts.
Alternatively, users can forgo `HPUPrecisionPlugin` and use only `torch.autocast` to control precision for every Op.
For nested contexts, the scope of a given context and its `enabled` parameter determine whether mixed precision is enabled or disabled in that region.


.. code::python

    # Granular autocast control without HPUPrecisionPlugin
    def forward(self, x):
        """Forward."""
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            torch.hpu.is_autocast_hpu_enabled() # Returns True

            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=False):
                torch.hpu.is_autocast_hpu_enabled() # Returns False

            # Re-entering autocast enabled region
            torch.hpu.is_autocast_hpu_enabled() # Returns True
        return super().forward(x)

    # Granular autocast control with HPUPrecisionPlugin
    def forward(self, x):
        """Forward."""
        # HPUPrecisionPlugin wraps a forward_context on train / val / predict / test _steps.
        # This makes torch.autocast(enabled=True) as used in previous example redundant.
        torch.hpu.is_autocast_hpu_enabled() # Returns True

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=False):
            torch.hpu.is_autocast_hpu_enabled() # Returns False
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                torch.hpu.is_autocast_hpu_enabled() # Returns True
            torch.hpu.is_autocast_hpu_enabled() # Returns False

        # Re-entering autocast enabled region
        torch.hpu.is_autocast_hpu_enabled() # Returns True
        return super().forward(x)


For more details, please refer to
`Native PyTorch Autocast <https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/Autocast.html>`__.
and `Automatic Mixed Precision Package: torch.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`__.

----

fp8 Training
-------------

Lightning supports fp8 training using HPUPrecisionPlugin, :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin`.

fp8 training is only available on Gaudi2 and above. Output from fp8 supported modules is in `torch.bfloat16`.

For fp8 training, call plugin.convert_modules(). The function accepts following args for the fp8 training:

1. `replace_layers` : Set `True` to let the plugin replace `torch.nn.Modules` with `transformer_engine` equivalent modules. You can directly import and use modules from `transformer_engine` as well.
2. `recipe` : fp8 recipe used in training.

.. code-block:: python

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe

    model = BoringModel()

    # init the precision plugin for fp8 training.
    plugin = HPUPrecisionPlugin(precision="fp8")

    # Replace torch.nn.Modules with transformer engine equivalent modules
    plugin.convert_modules(model, replace_layers=True, recipe=recipe.DelayedScaling())

    # Initialize a trainer with HPUPrecisionPlugin
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        plugins=plugin
    )

    # Train the model ⚡
    trainer.fit(model)


.. note::

    To use `transformer_engine` directly for training:

    1. Import `transformer_engine` and replace your modules with `transformer_engine` modules in the model.
    2. Wrap the forward pass of the training with `fp8_autocast`.

    Users may still use `HPUPrecisionPlugin` to train in mixed precision for modules not supported by `transformer_engine`.


.. note::

    To enable fp8 training with HPUDeepSpeedStrategy, use HPUDeepSpeedPrecisionPlugin, instead of HPUPrecisionPlugin, while keeping all other steps the same.

For more details on `transformer_engine` and `recipes`, refer to `FP8 Training with Intel Gaudi Transformer Engine <https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html>`__.


----

fp8 Inference
--------------

Lightning supports fp8 inference using HPUPrecisionPlugin, :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin`. fp8 inference is only available on Gaudi2 and above.

`Intel Neural Compressor` (INC) is required to run fp8 inference.

.. code-block:: bash

    python -um pip install neural-compressor


**Measurement and Quantization mechanisms**

Inference in fp8 is a two step process.

1. Measurement mode: This step injects PyTorch measurement hooks to the model. Model is run on a portion of the dataset, and these measurement hooks measure the data statistics (e.g. max abs) and outputs them at the filepath specified by json.

2. Quantization mode: This is achieved by replacing modules with quantized modules implemented in INC that quantize and dequantize the tensors. It includes multiple steps, viz:

   * Loading the measurements file.
   * Calculating the scale of each tensor from its measurement.
   * Injecting scale and cast ops to the model around ops that were selected to run in FP8.


**Measurement**

Get measurement data by running inference on a portion on data with `HPUPrecisionPlugin.convert_modules(model, inference=True, quant=False)`.


.. code-block:: python

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe

    model = BoringModel()

    # init the precision plugin for fp8 inference.
    plugin = HPUPrecisionPlugin(precision="fp8")

    # Replace module for fp8 inference measurements
    plugin.convert_modules(model, inference=True, quant=False)

    # Initialize a trainer with HPUPrecisionPlugin
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        plugins=plugin,
        limit_predict_batches=0.1,
    )

    # Run inference and dump measurements ⚡
    trainer.predict(model)


**Quantization**

Run inference with `HPUPrecisionPlugin.convert_modules(model, inference=True, quant=True)`.


.. code-block:: python

    # Replace module for fp8 inference measurements
    plugin.convert_modules(model, inference=True, quant=True)

    # Run inference ⚡
    trainer.predict(model)


**JSONs for quant and measure modes**

INC uses configuration jsons for selecting between quant and measurement modes.
This can be toggled via `quant` param in `HPUPrecisionPlugin.convert_modules()`.
`quant` also accepts user defined config dictionary or a json path.

Refer to `Supported JSON Config File Options <https://docs.habana.ai/en/v1.19.0/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#supported-json-config-file-options>`__ for more information on supported json configs.


.. note::

    To enable fp8 inference with HPUDeepSpeedStrategy, use HPUDeepSpeedPrecisionPlugin, instead of HPUPrecisionPlugin, while keeping all other steps the same.


For more details, refer to `Inference Using FP8 <https://docs.habana.ai/en/v1.19.0/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html>`__.
For a list of data types supported with HPU, refer to `PyTorch Support Matrix <https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Support_Matrix.html>`__.

----

Enabling DeviceStatsMonitor with HPUs
----------------------------------------

:class:`~lightning.pytorch.callbacks.device_stats_monitor.DeviceStatsMonitor` is a callback that automatically monitors and logs device stats during the training stage.
This callback can be passed for training with HPUs. It returns a map of the following metrics with their values in bytes of type uint64:

+-------------------+---------------------------------------------+
| Metric            | Value                                       |
+===================+=============================================+
| Limit             | Amount of total memory on HPU.              |
+-------------------+---------------------------------------------+
| InUse             | Amount of allocated memory at any instance. |
+-------------------+---------------------------------------------+
| MaxInUse          | Amount of total active memory allocated.    |
+-------------------+---------------------------------------------+
| NumAllocs         | Number of allocations.                      |
+-------------------+---------------------------------------------+
| NumFrees          | Number of freed chunks.                     |
+-------------------+---------------------------------------------+
| ActiveAllocs      | Number of active allocations.               |
+-------------------+---------------------------------------------+
| MaxAllocSize      | Maximum allocated size.                     |
+-------------------+---------------------------------------------+
| TotalSystemAllocs | Total number of system allocations.         |
+-------------------+---------------------------------------------+
| TotalSystemFrees  | Total number of system frees.               |
+-------------------+---------------------------------------------+
| TotalActiveAllocs | Total number of active allocations.         |
+-------------------+---------------------------------------------+


The below shows how ``DeviceStatsMonitor`` can be enabled.

.. code-block:: python

    from lightning import Trainer
    from lightning.callbacks import DeviceStatsMonitor
    from lightning_habana.pytorch.accelerator import HPUAccelerator

    device_stats = DeviceStatsMonitor()
    trainer = Trainer(accelerator=HPUAccelerator(), callbacks=[device_stats])

For more details, please refer to `Memory Stats APIs <https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/Python_Packages.html#memory-stats-apis>`__.


----

Runtime Environment Variables
----------------------------------------

Habana runtime environment flags are used to change the behavior as well as enable or disable some features.

For more information, refer to `Runtime Flags <https://docs.habana.ai/en/latest/PyTorch/Runtime_Flags.html#pytorch-runtime-flags>`__.


----

Using LightningCLI
-------------------

LightningCLI supports HPU. Following configurations from Lightning Habana are supported:

* accelerator: "auto", "hpu".
* strategies: "auto", "hpu_single", "hpu_parallel".
* plugins: class instances of `HPUPrecisionPlugin` and `HPUCheckpointIO`.

Limitations with HPU
^^^^^^^^^^^^^^^^^^^^^

* LightningCLI cannot use class instances of accelerator and strategies. `#19682 <https://github.com/Lightning-AI/pytorch-lightning/issues/19682>`__. Applies to Lightning accelerator and strategies as well.
* `HPUProfiler` does not work with LightningCLI since it is unable to patch `torch.profiler.ProfilerActivity` list.
