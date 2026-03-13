<div align="center">

# Discretax - State Space Models in JAX

<img src="https://raw.githubusercontent.com/camail-official/discretax/main/assets/logo.png" alt="Discretax logo" width="200"/>


[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/camail-official/discretax/actions/workflows/tests.yml/badge.svg)](https://github.com/camail-official/discretax/actions/workflows/test.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/camail-official/discretax/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)[![JAX](https://img.shields.io/badge/JAX-0.7%2B-0A7AAA?logo=jax&logoColor=white)](https://github.com/google/jax)
[![PyPI version](https://img.shields.io/pypi/v/discretax)](https://pypi.org/project/discretax/)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/VazrGCxeT7)

</div>

[discretax](https://github.com/camail-official/discretax) is a collection of state space models implemented in JAX. It is

- easy to use
- fast
- modular

## Table of contents

- [Discretax - State Space Models in JAX](#discretax---state-space-models-in-jax)
  - [Table of contents](#table-of-contents)
  - [News](#news)
  - [Just get me Going](#just-get-me-going)
  - [Join the Community](#join-the-community)
  - [Installation](#installation)
    - [Full Library Installation](#full-library-installation)
  - [Supported Models](#supported-models)
  - [Contributing](#contributing)
  - [Core Contributors](#core-contributors)
  - [Citation](#citation)

## News

- [2026-03]: After a big refactor, we are renaming the project from linax to discretax.
- [2025-10]: We are happy to launch the first beta version of linax. 🎉

## Just get me Going

If you don't care about the details, we provide [example notebooks](examples/) that are ready to use.

## Join the Community

To join our growing community of JAX and state space model enthusiasts, join our [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/VazrGCxeT7) server. Feel free to write us a message (either there or to our personal email, see the bottom of this page) if you have any questions, comments, or just want to say hi!

## Installation

[discretax](https://github.com/camail-official/discretax) is available as a PyPI package. To install it via uv, just run

```bash
uv add discretax
```

or

```bash
uv add discretax[cu12]
```

If pip is your package manager of choice, run

```bash
pip install discretax
```

or

```bash
pip install discretax[cu12]
```

### Full Library Installation

If you want to install the full library, especially if you want to **contribute** to the project, clone the [discretax](https://github.com/camail-official/discretax) repository and cd into it

```bash
git clone https://github.com/camail-official/discretax.git
cd discretax
```

If you want to install dependencies for CPU, run

```bash
uv sync
```

for GPU run

```bash
uv sync --extra cu12
```

To include development tooling (pre-commit, Ruff), install:

```bash
uv sync --extra dev
```

To install the packaged experiment and training stack (configs, datasets, Optax runtime, W&B logging), run:

```bash
uv sync --extra train
```

For NVIDIA GPU training, install the training stack together with CUDA-enabled JAX:

```bash
uv sync --extra train --extra cu12
```

Then verify that JAX sees the GPU:

```bash
uv run python scripts/env/check_jax_backend.py --require-gpu
```

After installing the development dependencies (activate your environment if needed), enable the git hooks:

```bash
uv run pre-commit install
```

## Experiment Training

The repository now includes a packaged training stack with:

- hierarchical YAML experiment configs under `configs/`
- dataset loaders for MNIST, CIFAR-10, and preprocessed UEA datasets
- an Optax-based Equinox training runtime with checkpoints, JSONL history, and run metadata
- optional Weights & Biases logging with stable run naming and flattened config logging

Helper scripts now live under `scripts/`:

- `scripts/env/` for environment and runtime checks
- `scripts/datasets/` for dataset download and preprocessing utilities

Print a fully resolved config:

```bash
uv run discretax-train --config configs/experiments/mnist_linoss_smoke.yaml --print-config
```

Run a configured experiment:

```bash
uv run discretax-train --config configs/experiments/mnist_linoss_smoke.yaml
```

Override config values from the command line with dotted assignments:

```bash
uv run discretax-train \
  --config configs/experiments/mnist_linoss_smoke.yaml \
  --set trainer.max_steps=10 \
  --set optimizer.learning_rate=0.001
```

Resume training from an existing run directory:

```bash
uv run discretax-train \
  --config configs/experiments/uea_eigenworms_linoss_sanity.yaml \
  --resume-from outputs/20260313-025347-uea-eigenworms-linoss-sanity \
  --set trainer.max_steps=32
```

Evaluate a saved checkpoint without further training:

```bash
uv run discretax-train \
  --config configs/experiments/uea_eigenworms_linoss_sanity.yaml \
  --resume-from outputs/20260313-025347-uea-eigenworms-linoss-sanity \
  --eval-only
```

Each run directory now includes:

- `config.yaml` with the fully resolved experiment config
- `history.jsonl` with step and evaluation metrics
- `summary.json` with final metrics and run timing
- `run_metadata.json` with git state, host info, JAX backend, and visible devices
- `checkpoints/` with `best`, `latest`, and periodic `step-*` snapshots

UEA support expects the preprocessed split layout used in the sibling `linoss` repositories:

```text
<data_root>/processed/UEA/<dataset_name>/X_train.pkl
<data_root>/processed/UEA/<dataset_name>/y_train.pkl
<data_root>/processed/UEA/<dataset_name>/X_val.pkl
<data_root>/processed/UEA/<dataset_name>/y_val.pkl
<data_root>/processed/UEA/<dataset_name>/X_test.pkl
<data_root>/processed/UEA/<dataset_name>/y_test.pkl
```

You can create that layout in this repo with:

```bash
uv sync --extra data
uv run python scripts/datasets/download_uea.py
uv run python scripts/datasets/process_uea.py --dataset EigenWorms
```

## Supported Models

| Year | Model  | Paper                                                                                                                                                                                         | Code                                                                      | Our implementation                                                                                       |
| ---- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 2024 | DeltaNet | [Parallelizing Linear Transformers with the Delta Rule](https://openreview.net/forum?id=y8Rm4VNRPH)                                                                                                    | [sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/deltanet.py)    |
| 2024 | LinOSS | [Oscillatory State Space Models](https://openreview.net/pdf?id=GRMfXcAAFh)                                                                                                                    | [tk-rusch/linoss](https://github.com/tk-rusch/linoss)                     | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/linoss.py)       |
| 2023 | LRU    | [Resurrecting Recurrent Neural Networks for Long Sequences](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)                                                                     | [LRU paper](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf) | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/lru.py)          |
| 2022 | S5     | [Simplified State Space Layers for Sequence Modeling](https://openreview.net/pdf?id=Ai8Hw3AXqks)                                                                                              | [lindermanlab/S5](https://github.com/lindermanlab/S5)                     | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/s5.py)           |
| 2022 | S4D    | [On the Parameterization and Initialization of Diagonal State Space Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/e9a32fade47b906de908431991440f7c-Paper-Conference.pdf) | [state-spaces/s4](https://github.com/state-spaces/s4)                     | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/sequence_mixers/s4d.py) |

## Contributing

If you want to contribute to the project, please check out [contributing](docs/contributing.md)

## Core Contributors

This repository has been created and is maintained by:

- [Philipp Nazari](https://phnazari.github.io)
- [Francesco Maria Ruscio](https://github.com/francescoshox)
- [Benedict Armstrong](https://github.com/benedict-armstrong)

This work has been carried out within the [Computational Applied Mathematics & AI Lab](https://camail.org),
led by [T. Konstantin Rusch](https://github.com/tk-rusch).

## Citation

If you find this repository useful, please consider citing it.

```bib
@software{discretax2025,
  title  = {Discretax: A Lightweight Collection of State Space Models in JAX},
  author = {Nazari, Philipp* and Ruscio, Francesco Maria* and Armstrong, Benedict* and Rusch, T. Konstantin},
  url    = {https://github.com/camail-official/discretax},
  year   = {2025}
}
```
