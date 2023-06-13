:orphan:

.. _hpu_advanced:

Accelerator: HPU Training
=========================
This document offers instructions to Gaudi chip users who want to use advanced strategies and profiling HPUs.

----

Using HPUProfiler
-----------------

HPUProfiler is a Lightning implementation of PyTorch profiler for HPU. It aids in obtaining profiling summary of PyTorch functions.
It subclasses PyTorch Lightning's `PyTorch profiler <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.profilers.PyTorchProfiler.html#pytorch_lightning.profilers.PyTorchProfiler>`_.

Default Profiling
^^^^^^^^^^^^^^^^^^
For auto profiling, create an ``HPUProfiler`` instance and pass it to the trainer.
At the end of ``profiler.fit()``, it will generate a JSON trace for the run.
In case ``accelerator= HPUAccelerator()`` is not used with ``HPUProfiler``, it will dump only CPU traces, similar to ``PyTorchProfiler``.

.. code-block:: python

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    trainer = Trainer(accelerator=HPUAccelerator(), profiler=HPUProfiler())

Distributed Profiling
^^^^^^^^^^^^^^^^^^^^^^

To profile a distributed model, use ``HPUProfiler`` with the filename argument which will save a report per rank.

.. code-block:: python

    from pytorch_lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    profiler = HPUProfiler(filename="perf-logs")
    trainer = Trainer(profiler=profiler, accelerator=HPUAccelerator())

Custom Profiling
^^^^^^^^^^^^^^^^^

To `profile custom actions of interest <https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_expert.html#profile-custom-actions-of-interest>`_,
reference a profiler in the ``LightningModule``.

.. code-block:: python

    from pytorch_lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
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
    trainer = Trainer(accelerator=HPUAccelerator(), profiler=profiler)

For more details on Profiler, refer to `PyTorchProfiler <https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_intermediate.html>`_

Visualizing Profiled Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiler dumps traces in JSON format. The traces can be visualized in 2 ways as described below.

Using PyTorch TensorBoard Profiler
""""""""""""""""""""""""""""""""""

For further instructions see, https://github.com/pytorch/kineto/tree/master/tb_plugin.

1. Install tensorboard

.. code-block:: bash

    python -um pip install tensorboard torch-tb-profiler

2. Start the TensorBoard server (default at port 6006)

.. code-block:: bash

    tensorboard --logdir ./tensorboard --port 6006

3. Open the following URL in your browser: `http://localhost:6006/#profile`.

Using Chrome
"""""""""""""

    1. Open Chrome and paste this URL: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.

Limitations
^^^^^^^^^^^^

- When using ``HPUProfiler``, wall clock time will not be representative of the true wall clock time. This is due to forcing profiled operations to be measured synchronously, when many HPU ops happen asynchronously.
  It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use the ``SimpleProfiler``.

- ``HPUProfiler.summary()`` is not supported.

- Passing the Profiler name as a string "hpu" to the trainer is not supported.

----

Using DeepSpeed
------------------------

HPU supports advanced strategies like ``deepspeed``. By default, HPU training uses 32-bit precision.
To enable mixed precision, set the ``precision`` flag.

.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

For further details on the supported DeepSpeed features and functionalities, refer to `Using Deepspeed with HPU <https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/index.html>`_.
