# Lightning ⚡ Intel Habana

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI Status](https://badge.fury.io/py/lightning-habana.svg)](https://badge.fury.io/py/lightning-habana)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-habana)](https://pypi.org/project/lightning-habana/)
[![PyPI Status](https://pepy.tech/badge/lightning-habana)](https://pepy.tech/project/lightning-habana)
[![Deploy Docs](https://github.com/Lightning-AI/lightning-Habana/actions/workflows/docs-deploy.yml/badge.svg)](https://lightning-ai.github.io/lightning-Habana/)

[![General checks](https://github.com/Lightning-AI/lightning-habana/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-habana/actions/workflows/ci-checks.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status/Lightning-AI.lightning-Habana?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=45&branchName=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-Habana/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-Habana/main)

[Habana® Gaudi® AI Processor (HPU)](https://habana.ai/) training processors are built on a heterogeneous architecture with a cluster of fully programmable Tensor Processing Cores (TPC) along with its associated development tools and libraries, and a configurable Matrix Math engine.

The TPC core is a VLIW SIMD processor with an instruction set and hardware tailored to serve training workloads efficiently.
The Gaudi memory architecture includes on-die SRAM and local memories in each TPC and,
Gaudi is the first DL training processor that has integrated RDMA over Converged Ethernet (RoCE v2) engines on-chip.

On the software side, the PyTorch Habana bridge interfaces between the framework and SynapseAI software stack to enable the execution of deep learning models on the Habana Gaudi device.

Gaudi provides a significant cost-effective benefit, allowing you to engage in more deep learning training while minimizing expenses.

For more information, check out [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Overview.html#gaudi-architecture%3E) and [Gaudi Developer Docs](https://developer.habana.ai).

______________________________________________________________________

## Installing Lighting Habana

To install Lightning Habana, run the following command:

```bash
pip install -U lightning lightning-habana
```

## Using PyTorch Lighting with HPU

To enable PyTorch Lightning with HPU accelerator, provide `accelerator=HPUAccelerator()` parameter to the Trainer class.

```python
from lightning import Trainer
from lightning_habana.pytorch.accelerator import HPUAccelerator

# Run on as many HPUs as available by default.
trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
# Equivalent to
trainer = Trainer()

# Run on one HPU.
trainer = Trainer(accelerator=HPUAccelerator(), devices=1)
# Run on multiple HPUs.
trainer = Trainer(accelerator=HPUAccelerator(), devices=8)
# Choose the number of devices automatically.
trainer = Trainer(accelerator=HPUAccelerator(), devices="auto")
```

The `devices=1` parameter with HPUs enables the Habana accelerator for single card training using `SingleHPUStrategy`.

The `devices>1` parameter with HPUs enables the Habana accelerator for distributed training. It uses `HPUParallelStrategy` which is based on DDP strategy with the integration of Habana’s collective communication library (HCCL) to support scale-up within a node and scale-out across multiple nodes.


# Support Matrix

    +------------------------+----------------------------------------------------------+
    | **SynapseAI**          |            1.10.0                                        |
    +========================+==========================================================+
    | **PyTorch**            | 2.0.1                                                    |
    |                        |                                                          |
    +------------------------+----------------------------------------------------------+
    | **PyTorch Lightning**  | 2.0.0                                                    |
    |          or            |                                                          |
    | **Lightning**          | 2.0.0                                                    |
    |                        |                                                          |
    +------------------------+----------------------------------------------------------+
    | **Lightning Habana**   | 1.0.0                                                    |
    |                        |                                                          |
    +------------------------+----------------------------------------------------------+
    | **DeepSpeed**          | Forked from v0.7.7 of the official                       |
    |                        | DeepSpeed.                                               |
    |                        |                                                          |
    +------------------------+----------------------------------------------------------+

For more information, check out [HPU Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html)

