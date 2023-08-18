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

    from lightning import Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler

    profiler = HPUProfiler(filename="perf-logs")
    trainer = Trainer(profiler=profiler, accelerator=HPUAccelerator())

Custom Profiling
^^^^^^^^^^^^^^^^^

To `profile custom actions of interest <https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_expert.html#profile-custom-actions-of-interest>`_,
reference a profiler in the ``LightningModule``.

.. code-block:: python

    from lightning import Trainer
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

HPU supports advanced optimization libraries like ``deepspeed``. The HabanaAI GitHub has a fork of the DeepSpeed library that includes changes to add support for SynapseAI.


Installing DeepSpeed for HPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use DeepSpeed with Lightning on Gaudi, you must install Habana's fork for DeepSpeed.
To install the latest supported version of DeepSpeed, follow the instructions at https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html#installing-deepspeed-library


Using DeepSpeed on HPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Lightning, Deepspeed functionalities are enabled for HPU via HPUDeepSpeedStrategy. By default, HPU training uses 32-bit precision. To enable mixed precision, set the ``precision`` flag.
A basic example of HPUDeepSpeedStrategy invocation is shown below.

.. code-block:: python

    class DemoModel(LightningModule):

        ...

        def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[_TORCH_LRSCHEDULER]]:
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = DemoModel()
    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    trainer = Trainer(
        accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(),
        callbacks=[TestCB()], max_epochs=1, plugins=_plugins,
    )
    trainer.fit(model)

.. note::

   1. accelerator="auto" or accelerator="hpu" is not yet enabled with lightning>2.0.0 and lightning-habana.
   2. Passing strategy in a string representation ("hpu_deepspeed", "hpu_deepspeed_stage_1", etc.. ) are not yet enabled.

DeepSpeed Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^

Below is a summary of all the DeepSpeed configurations supported by HPU. For full details on the HPU supported DeepSpeed features and functionalities, refer to `Using Deepspeed with HPU <https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/index.html>`_.
All further information on DeepSpeed configurations can be found in DeepSpeed<https://www.deepspeed.ai/training/#features> documentation.

* ZeRO-1

* ZeRO-2

* ZeRO-3

* ZeRO-Offload

* ZeRO-Infinity

* BF16 precision

* BF16Optimizer

* Activation Checkpointing

The HPUDeepSpeedStrategy can be configured using its arguments or a JSON configuration file. Both configuration methods are shown in the examples below.

ZeRO-1
""""""
.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(zero_optimization=True, stage=1), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

ZeRO-2
""""""
.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(zero_optimization=True, stage=2), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

ZeRO-3
""""""
.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(zero_optimization=True, stage=3), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

ZeRO-Offload
""""""""""""
.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(zero_optimization=True, stage=2, offload_optimizer=True), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

ZeRO-Infinity
""""""""""""""
.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(zero_optimization=True, stage=2, offload_optimizer=True), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

BF16 precision
""""""""""""""

.. code-block:: python

    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    trainer = Trainer(devices=8, accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(), plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")])

BF16-Optimizer
""""""""""""""
This example demonstrates how the HPUDeepSpeedStrategy can be configured using a DeepSpeed json configuration.

.. code-block:: python

    from lightning.pytorch import LightningModule, Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy

    config = {
        "train_batch_size": 8,
        "bf16": {
            "enabled": True
        },
        "fp16": {
            "enabled": False
        },
        "train_micro_batch_size_per_gpu": 2,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
            "warmup_min_lr": 0.02,
            "warmup_max_lr": 0.05,
            "warmup_num_steps": 4,
            "total_num_steps" : 8,
            "warmup_type": "linear"
            }
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {"stage" : 2}
    }


    class SampleModel(LightningModule):
        ...

        def configure_optimizers(self):
            from torch.optim.adamw import AdamW as AdamW
            optimizer = torch.optim.AdamW(self.parameters())
            return optimizer


    _plugins = [DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
    _accumulate_grad_batches=2
    _parallel_hpus = [torch.device("hpu")] * HPUAccelerator.auto_device_count()

    model = SampleModel()
    trainer = Trainer(
        accelerator=HPUAccelerator(), strategy=HPUDeepSpeedStrategy(config=config, parallel_devices=_parallel_hpus),
        enable_progress_bar=False,
        fast_dev_run=8,
        plugins=_plugins,
        use_distributed_sampler=False,
        limit_train_batches=16,
        accumulate_grad_batches=_accumulate_grad_batches,
    )

    trainer.fit(model)

.. note::

   1. When the optimizer and/or scheduler configuration is specified in both LightningModule and DeepSpeed json configuration file, preference will be given to the optimizer/scheduler returned by LightningModule::configure_optimizers().


Activation Checkpointing
""""""""""""""""""""""""

.. code-block:: python

    from lightning.pytorch import LightningModule, Trainer
    from lightning_habana.pytorch.accelerator import HPUAccelerator
    from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

    class SampleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(32)
            self.l2 = nn.Linear(32)

        def forward(self, x):
            l1_out = self.l1(x)
            l2_out = checkpoint(self.l2, l1_out)
            return l2_out

    trainer = Trainer(accelerator=HPUAccelerator(),
                        strategy=HPUDeepSpeedStrategy(zero_optimization=True,
                                                        stage=3,
                                                        offload_optimizer=True,
                                                        cpu_checkpointing=True),
                        plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")]
                    )

Limitations of DeepSpeed on HPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   1. DeepSpeed Zero Stage 3 is not yet supported by Gaudi2.
   2. Offloading to Nvme is not yet verified on HPU with DeepSpeed Zero Stage 3 Offload configuration.
   3. Model Pipeline and Tensor Parallelism are currently supported only on Gaudi2.
