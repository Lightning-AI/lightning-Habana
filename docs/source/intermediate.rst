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
These also allow for fine tuning with `enabled` for enabling and disabling mixed precision training for certain parts of the code.

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

For more details, please refer to
`Native PyTorch Autocast <https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/Autocast.html>`__.
and `Automatic Mixed Precision Package: torch.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`__.

----

fp8 Training
----------------------------------------

Lightning supports fp8 training using HPUPrecisionPlugin, :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin`.
fp8 training is only available on Gaudi2 and above. Output from fp8 supported modules is in `torch.bfloat16`.

The plugin accepts following args for the fp8 training:

1. `replace_layers` : Set `True` to let the plugin replace `torch.nn.Modules` with `trandformer_engine` equivalent modules. You can directly import and use modules from `transformer_engine` as well.

2. `recipe` : fp8 recipe used in training.

.. code-block:: python

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe

    model = BoringModel()

    # init the precision plugin for fp8 training.
    plugin = HPUPrecisionPlugin(precision="fp8", replace_layers=True, recipe=recipe.DelayedScaling())

    # Replace torch.nn.Modules with transformer engine equivalent modules
    plugin.convert_modules(model)

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

    Users may still use `HPUPrecisionPlugin` to train in `bf16-mixed` precision for modules not supported by `transformer_engine`.

For more details on `transformer_engine` and `recipes`, refer to `FP8 Training with Intel Gaudi Transformer Engine <https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html>`__.


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
