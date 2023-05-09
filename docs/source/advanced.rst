:orphan:

.. _hpu_advanced:

Accelerator: HPU training
=========================
**Audience:** Gaudi chip users looking to use advanced strategies and profiling HPU's.

----

Working with HPUProfiler
-------------------------

HPUProfiler is a lightning implementation of PyTorch profiler for HPU devices. It aids in obtaining profiling summary of PyTorch functions.
It subclasses PyTorch Lightning's `PyTorch profiler <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.profilers.PyTorchProfiler.html#pytorch_lightning.profilers.PyTorchProfiler>`_.

Default Profiling
^^^^^^^^^^^^^^^^^^
For auto profiling, create a HPUProfiler instance and pass it to trainer.
At the end of ``profiler.fit()``, it will generate a json trace for the run.
In case ``accelerator= HPUAccelerator()`` is not used with HPUProfiler, then it will dump only CPU traces, similar to PyTorchProfiler.

.. code-block:: python

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    trainer = Trainer(accelerator=HPUAccelerator(), profiler=HPUProfiler())

Distributed Profiling
^^^^^^^^^^^^^^^^^^^^^^

To profile a distributed model, use the HPUProfiler with the filename argument which will save a report per rank:

.. code-block:: python

    from pytorch_lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    profiler = HPUProfiler(filename="perf-logs")
    trainer = Trainer(profiler=profiler, accelerator=HPUAccelerator())

Custom Profiling
^^^^^^^^^^^^^^^^^

To `profile custom actions of interest <https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_expert.html#profile-custom-actions-of-interest>`_, reference a profiler in the ``LightningModule``.:

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

For more details on profiler, refer to `PyTorchProfiler <https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_intermediate.html>`_

Visualize Profiled Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiler will dump traces in json format. The traces can be visualized in 2 ways:

Using PyTorch TensorBoard Profiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For further instructions see, https://github.com/pytorch/kineto/tree/master/tb_plugin.

Install tensorboard
"""""""""""""""""""""
.. code-block:: bash

    python -um pip install tensorboard torch-tb-profiler

Start the TensorBoard server (default at port 6006)
""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    tensorboard --logdir ./tensorboard --port 6006

Now open the following url in your browser
""""""""""""""""""""""""""""""""""""""""""""
 http://localhost:6006/#profile


Using Chrome
^^^^^^^^^^^^^

    1. Open Chrome and copy/paste this URL: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.

Limitations
^^^^^^^^^^^^

- When using the HPUProfiler, wall clock time will not be representative of the true wall clock time. This is due to forcing profiled operations to be measured synchronously, when many HPU ops happen asynchronously.
  It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use the SimpleProfiler.

- ``HPUProfiler.summary()`` is not supported

- Passing profiler name as string "hpu" to the trainer is not supported.

----

Working with DeepSpeed
------------------------

HPU's support advanced strategies like ``deepspeed``.
By default, HPU training will use 32-bit precision. To enable mixed precision, set the ``precision`` flag.

.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=1, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

More details on the HPU supported deepspeed features and functionalities, refer to refer to `Deepspeed with HPU <https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/index.html>`_
