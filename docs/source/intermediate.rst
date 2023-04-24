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

    trainer = Trainer(devices=1, accelerator="hpu", precision=16)

----

Customize Mixed Precision
-------------------------

Internally, :class:`~pytorch_lightning.plugins.precision.hpu.HPUPrecisionPlugin` uses the Habana Mixed Precision (HMP) package to enable mixed precision training.

You can execute the ops in FP32 or BF16 precision. The HMP package modifies the Python operators to add the appropriate cast operations for the arguments before execution.
The default settings enable users to enable mixed precision training with minimal code easily.

In addition to the default settings in HMP, users also have the option of overriding these defaults and providing their
BF16 and FP32 operator lists by passing them as parameter to :class:`~pytorch_lightning.plugins.precision.hpu.HPUPrecisionPlugin`.

The below snippet shows an example model using MNIST with a single Habana Gaudi device and making use of HMP by overriding the default parameters.
This enables advanced users to provide their own BF16 and FP32 operator list instead of using the HMP defaults.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.plugins import HPUPrecisionPlugin

    # Initialize a trainer with HPU accelerator for HPU strategy for single device,
    # with mixed precision using overidden HMP settings
    trainer = pl.Trainer(
        accelerator="hpu",
        devices=1,
        # Optional Habana mixed precision params to be set
        # Checkout `examples/pl_hpu/ops_bf16_mnist.txt` for the format
        plugins=[
            HPUPrecisionPlugin(
                precision=16,
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

    device_stats = DeviceStatsMonitor()
    trainer = Trainer(accelerator="hpu", callbacks=[device_stats])

For more details, please refer to `Memory Stats APIs <https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/Python_Packages.html#memory-stats-apis>`__.

----

Working with HPUProfiler
-------------------------

HPUProfiler is a lightning implementation of PyTorch profiler for HPU devices. It aids in obtaining profiling summary of PyTorch functions. 
It subclasses PyTorch Lightning's [PyTorch profiler](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.profilers.PyTorchProfiler.html#pytorch_lightning.profilers.PyTorchProfiler).

Default Profiling
^^^^^^^^^^^^^^^^^^
For auto profiling, create a HPUProfiler instance and pass it to trainer.
At the end of ``profiler.fit()``, it will generate a json trace for the run.
In case ``accelerator = "hpu"`` is not used with HPUProfiler, then it will dump only CPU traces, similar to PyTorchProfiler.

.. code-block:: python
    # Import profiler
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    # Create profiler object
    profiler = HPUProfiler()
    accelerator = "hpu"

    # Pass profiler to the trainer
        trainer = Trainer(
            profiler=profiler,
            accelerator=accelerator,
        )

Distributed Profiling
^^^^^^^^^^^^^^^^^^^^^^

To profile a distributed model, use the HPUProfiler with the filename argument which will save a report per rank:

.. code-block:: python
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    profiler = HPUProfiler(filename="perf-logs")
    trainer = Trainer(profiler=profiler, accelerator="hpu")

Custom Profiling
^^^^^^^^^^^^^^^^^

To [profile custom actions of interest](https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_expert.html#profile-custom-actions-of-interest), reference a profiler in the LightningModule:

.. code-block:: python

    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    # Reference profiler in LightningModule
    class MyModel(LightningModule):
        def __init__(self, profiler=None):
            self.profiler = profiler

    # To profile in any part of your code, use the self.profiler.profile() function
        def custom_processing_step_basic(self, data):
            with self.profiler.profile("my_custom_action"):
                print("do somthing")
            return data

    # Alternatively, use self.profiler.start("my_custom_action")
    # and self.profiler.stop("my_custom_action") functions
    # to enclose the part of code to be profiled.
        def custom_processing_step_granular(self, data):
            self.profiler.start("my_custom_action")
            print("do somthing")
            self.profiler.stop("my_custom_action")
            return data

    # Pass profiler instance to LightningModule
    profiler = HPUProfiler()
    model = MyModel(profiler)
    trainer = Trainer(profiler=profiler, accelerator="hpu")

For more details on profiler, refer to [PyTorchProfiler](https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_intermediate.html)

Visualize Profiled Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiler will dump traces in json format. The traces can be visualized in 2 ways:

Using PyTorch TensorBoard Profiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For further instructions see, https://github.com/pytorch/kineto/tree/master/tb_plugin.

.. code-block:: bash
    # Install tensorboard
    python -um pip install tensorboard torch-tb-profiler

    # Start the TensorBoard server (default at port 6006):
    tensorboard --logdir ./tensorboard --port 6006

    # Now open the following url on your browser
    http://localhost:6006/#profile


Using Chrome
^^^^^^^^^^^^^

    1. Open Chrome and copy/paste this URL: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.

Limitations
^^^^^^^^^^^^

- When using the HPUProfiler, wall clock time will not be representative of the true wall clock time. This is due to forcing profiled operations to be measured synchronously, when many HPU ops happen asynchronously.
  It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use the SimpleProfiler.

- HPUProfiler.summary() is not supported

- Passing profiler name as string "hpu" to the trainer is not supported.
