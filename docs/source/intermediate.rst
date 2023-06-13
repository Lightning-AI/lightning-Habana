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

Customize Mixed Precision
-------------------------

Internally, :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin` uses the Habana Mixed Precision (HMP) package to enable mixed precision training.

You can execute the ops in FP32 or BF16 precision. The HMP package modifies the Python operators to add the appropriate cast operations for the arguments before execution.
With the default settings, you can easily enable mixed precision training with minimal code.

In addition to the default settings in HMP, you can choose to override these defaults and provide your own BF16 and FP32 operator lists by passing them as parameters
to :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin`.

The following is an excerpt from an MNIST example implemented on a single HPU.

.. code-block:: python

    import pytorch_lightning as pl
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin

    # Initialize a trainer with HPU accelerator for HPU strategy for single device,
    # with mixed precision using overidden HMP settings
    trainer = pl.Trainer(
        accelerator=HPUAccelerator(),
        devices=1,
        # Optional Habana mixed precision params to be set
        # Checkout `examples/pl_hpu/ops_bf16_mnist.txt` for the format
        plugins=[
            HPUPrecisionPlugin(
                precision="bf16-mixed",
                opt_level="O1",
                verbose=False,
                bf16_file_path="ops_bf16_mnist.txt",
                fp32_file_path="ops_fp32_mnist.txt",
            )
        ],
    )

    # Init our model
    model = LitClassifier()
    # Init the data
    dm = MNISTDataModule(batch_size=batch_size)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)

For more details, please refer to `PyTorch Mixed Precision Training on Gaudi <https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/PT_Mixed_Precision.html>`__.

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

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    from lightning_habana.pytorch.accelerator import HPUAccelerator

    device_stats = DeviceStatsMonitor()
    trainer = Trainer(accelerator=HPUAccelerator(), callbacks=[device_stats])

For more details, please refer to `Memory Stats APIs <https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/Python_Packages.html#memory-stats-apis>`__.
