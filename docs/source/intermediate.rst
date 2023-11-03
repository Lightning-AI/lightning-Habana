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
            with torch.autocast("device_type="hpu", dtype=torch.bfloat16):
            ...
            return

    class AutocastModelDecorator(nn.Module):
        # Autocast can be used as a decorator to the required code block.
        @torch.autocast("device_type="hpu", dtype=torch.bfloat16)
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

Enabling DeviceStatsMonitor with HPUs
----------------------------------------

:class:`~pytorch_lightning.callbacks.device_stats_monitor.DeviceStatsMonitor` is a callback that automatically monitors and logs device stats during the training stage.
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


-------------------------------------

Enabling recipe caching with HPU
-------------------------------------

Recipe caching helps to reduce graph compilations when training on multiple HPUs.
This is helpful when dealing with models that have dynamicity. Graphs are compiled once and cached to a user-specified location where other HPU devices may reuse them.
Recipe caching is disabled by default. To enable recipe caching, se the `PT_HPU_RECIPE_CACHE_CONFIG` environment variable. 
Configuration is encoded as a comma separated list in the following format: ‘<RECIPE_CACHE_PATH>,<RECIPE_CACHE_DELETE>,<RECIPE_CACHE_SIZE_MB>’.:
1. <RECIPE_CACHE_PATH> - Path (directory), where compiled graph recipes are stored to accelerate a scale up scenario. Only one process compiles the recipe, and other processes read it from disk.
2. <RECIPE_CACHE_DELETE> - Bool flag (true/false). If set to True, the directory provided as <RECIPE_CACHE_PATH> will be cleared when the workload starts.
3. <RECIPE_CACHE_SIZE_MB> - Max size in MB of recipe cache directory. If size limit is reached then the oldest recipes (by creation time on file system) are removed.

For more information, refer to `Handling Dynamic Shapes <https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html?highlight=PT_HPU_METRICS_FILE#detecting-and-mitigating-dynamicity-overview>`__.
