:orphan:

.. _hpu_intermediate:

Accelerator: HPU training
=========================
**Audience:** Gaudi chip users looking to save memory and scale models with mixed-precision training.

----

Enable Mixed Precision
----------------------

Lightning also allows mixed precision training with HPUs.
By default, HPU training will use 32-bit precision. To enable mixed precision, set the ``precision`` flag.

.. code-block:: python

    from lightning_habana.pytorch.accelerator import HPUAccelerator

    trainer = Trainer(devices=1, accelerator=HPUAccelerator(), precision="bf16-mixed")

----

Customize Mixed Precision
-------------------------

Internally, :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin` uses the Habana Mixed Precision (HMP) package to enable mixed precision training.

You can execute the ops in FP32 or BF16 precision. The HMP package modifies the Python operators to add the appropriate cast operations for the arguments before execution.
The default settings enable users to enable mixed precision training with minimal code easily.

In addition to the default settings in HMP, users also have the option of overriding these defaults and providing their
BF16 and FP32 operator lists by passing them as parameter to :class:`~lightning_habana.pytorch.plugins.precision.HPUPrecisionPlugin`.

The below snippet shows an example model using MNIST with a single Habana Gaudi device and making use of HMP by overriding the default parameters.
This enables advanced users to provide their own BF16 and FP32 operator list instead of using the HMP defaults.

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

- **Limit**: amount of total memory on HPU device.
- **InUse**: amount of allocated memory at any instance.
- **MaxInUse**: amount of total active memory allocated.
- **NumAllocs**: number of allocations.
- **NumFrees**: number of freed chunks.
- **ActiveAllocs**: number of active allocations.
- **MaxAllocSize**: maximum allocated size.
- **TotalSystemAllocs**: total number of system allocations.
- **TotalSystemFrees**: total number of system frees.
- **TotalActiveAllocs**: total number of active allocations.

The below snippet shows how DeviceStatsMonitor can be enabled.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    from lightning_habana.pytorch.accelerator import HPUAccelerator

    device_stats = DeviceStatsMonitor()
    trainer = Trainer(accelerator=HPUAccelerator(), callbacks=[device_stats])

For more details, please refer to `Memory Stats APIs <https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/Python_Packages.html#memory-stats-apis>`__.
