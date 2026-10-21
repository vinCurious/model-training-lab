# Model Training Lab

> From first principles to modern LLM training and post-training.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-000000?logo=apple&logoColor=white)
![Progress](https://img.shields.io/badge/Current-Module%2004-2ea44f)

A hands-on model-training laboratory that starts with gradients and optimization, builds language modeling and Transformers from first principles, and then moves into modern LLM fine-tuning, parameter-efficient training, quantization, and evaluation.

The goal is not to treat deep-learning libraries as black boxes.

> **Build it, inspect it, break it, measure it, then abstract it.**

---

## What This Repository Explores

The project progresses through four layers:

```text
training mechanics
↓
neural-network trainability
↓
language modeling & Transformers
↓
modern LLM post-training
```

Experiments begin with deliberately small models where every tensor, gradient, and failure mode can be inspected.

Higher-level libraries and real pretrained models are introduced only after the underlying mechanics are understood.

---

## Progress

| Module | Focus | Status |
|:---:|---|:---:|
| `00` | Environment, PyTorch, MPS & reproducibility | ✅ |
| `01` | Training fundamentals from first principles | ✅ |
| `02` | Neural networks & deep-network trainability | ✅ |
| `03` | Language modeling & next-token prediction | ✅ |
| `04` | Transformer / GPT from scratch | 🚧 Current |
| `05` | Pretrained models & full fine-tuning | Planned |
| `06` | Supervised fine-tuning (SFT) | Planned |
| `07` | LoRA & parameter-efficient fine-tuning | Planned |
| `08` | Quantization, QLoRA & MLX | Planned |
| `09` | Evaluation & error analysis | Planned |
| `10` | End-to-end model-training capstone | Planned |

---

## Representative Experiments

Rather than only implementing the successful path, the notebooks intentionally create failure cases and inspect what happens.

| Experiment | Observation |
|---|---|
| Learning-rate sweep | A sufficiently large learning rate caused catastrophic divergence |
| Gradient accumulation | Forgetting to clear gradients changed the optimization behavior dramatically |
| Model capacity | Larger models reduced training loss but did not automatically generalize better |
| Tiny initialization | Deep ReLU activations and gradients collapsed |
| Large initialization | Activations, loss, and gradients exploded |
| `tanh` saturation | Saturation was visible without forcing a simplistic vanishing-gradient conclusion |
| Residual-only network | More parameters and residual connections did not automatically improve training |
| PreNorm residual network | Normalization + residual structure produced strong, stable optimization |
| Ambiguous next-token context | The model learned an approximately `50/50` conditional distribution |
| Autoregressive generation | Low training loss did not prevent degradation on model-generated contexts |
| Temperature scaling | Changed decoding sharpness without changing the trained model |

The goal is to preserve counterexamples rather than hide them when an experiment does not match a textbook narrative.

---

## Experimental Method

Most experiments follow the same loop:

```text
question
↓
hypothesis
↓
minimal implementation
↓
measure
↓
inspect
↓
break an assumption
↓
measure again
↓
interpret
```

Depending on the experiment, measurements include:

- training and validation loss
- tensor shapes
- activation statistics
- gradient magnitudes
- parameter counts
- optimization trajectories
- learned probability distributions
- generated outputs
- generalization behavior
- memory and compute requirements

Notebook outputs are retained when they form part of the experimental evidence.

---

## Repository Structure

```text
model-training-lab/
├── README.md
├── requirements.txt
│
├── module00_setup/
│   ├── system_check.py
│   └── notes.md
│
├── module01_training_fundamentals/
│   ├── 01_manual_linear_regression.ipynb
│   ├── 02_train_validation_overfitting.ipynb
│   └── notes.md
│
├── module02_neural_networks/
│   ├── 01_from_linear_to_mlp.ipynb
│   ├── 02_initialization_and_gradients.ipynb
│   ├── 03_training_deeper_networks.ipynb
│   └── notes.md
│
├── module03_language_modeling/
│   ├── 01_tokens_embeddings_and_context.ipynb
│   ├── 02_next_token_prediction.ipynb
│   ├── 03_generation_and_sampling.ipynb
│   └── notes.md
│
├── data/
├── outputs/
└── checkpoints/
```

Notebooks contain the experiment narrative, implementation, measurements, and interpretation.

Each module's `notes.md` is a compact technical reference rather than a duplicate of the notebook.

As the project grows, reusable training and evaluation logic will move into Python modules and scripts where that abstraction becomes useful.

---

## Tooling & ML Ecosystem

The project begins with minimal dependencies and introduces production and research tooling as the experiments require it.

### Core

- Python
- PyTorch
- Jupyter
- NumPy
- matplotlib
- Apple MPS

### Models, datasets & post-training

Later modules will use the Hugging Face ecosystem for real model-training workflows:

- **Datasets** — loading, preprocessing, and working with public datasets
- **Transformers** — pretrained model architectures and checkpoints
- **PEFT** — LoRA and parameter-efficient fine-tuning
- **TRL** — post-training workflows where appropriate
- **Hugging Face Hub** — publishing models, checkpoints, metadata, and model cards

The progression will intentionally move from:

```text
implement the mechanism ourselves
↓
understand its behavior
↓
use the standard library abstraction
```

rather than starting with high-level training APIs.

### Apple Silicon

For experiments where it is useful, the project will also explore:

- MLX
- MLX-LM

particularly for efficient local inference and fine-tuning on Apple Silicon.

### Experiment tracking & evaluation

As experiments become larger, the repository will introduce tooling such as:

- TensorBoard
- MLflow
- Weights & Biases where useful
- `lm-eval-harness` or equivalent evaluation tooling

The goal is to move beyond manually reading notebook loss values toward reproducible experiment comparison.

---

## Data

Early experiments use synthetic or deliberately tiny datasets because controlled data makes optimization behavior easy to inspect.

Later modules will transition to real public datasets and benchmarks.

Sources may include:

- Hugging Face Datasets
- canonical datasets released with papers
- public benchmark datasets
- task-specific open datasets

Dataset provenance, preprocessing decisions, train/validation/test splits, and evaluation methodology will be recorded as part of the experiment.

Large datasets and model checkpoints will not be committed directly to Git.

---

## Research Workflow

The later stages of the project will increasingly resemble a small research workflow:

```text
paper / technique
↓
understand the core idea
↓
build a minimal implementation
↓
reproduce expected behavior
↓
run controlled variations
↓
evaluate
↓
document conclusions
```

Primary sources such as papers, technical reports, arXiv, and OpenReview will be preferred when studying model architectures or training techniques.

The emphasis is on distinguishing:

```text
what the paper claims
what the implementation does
what the experiment actually shows
```

---

## Reproducibility

Experiments aim to make assumptions visible:

- random seeds are set when reproducibility matters
- tensor dimensions are kept explicit
- important hyperparameters are recorded
- training and validation behavior are both inspected
- important notebook outputs are committed
- checkpoints and large generated artifacts remain outside Git

A result should be understandable without relying on hidden notebook state.

---

## Environment

Current experiments are designed to run locally on Apple Silicon.

```text
Python 3.12
PyTorch
MPS
Jupyter
```

The project deliberately starts with local compute.

Cloud GPUs will be introduced only when model size or training cost makes local execution impractical, and experiments requiring paid compute will be identified explicitly.

---

## Current Work

### Module 04 — Transformer / GPT From Scratch

The next step replaces the fixed flattened-context representation from Module 03 with token-to-token interaction.

The progression will be approximately:

```text
token embeddings
↓
queries, keys, values
↓
scaled dot-product attention
↓
causal masking
↓
multi-head attention
↓
Transformer block
↓
decoder-only language model
↓
training
↓
autoregressive generation
```

The objective is to understand why the Transformer architecture solves limitations that were directly observable in the simpler language model.

---

## Principles

**Understand before abstracting.**

**Measure rather than assume.**

**Break things intentionally.**

**Keep tensor shapes explicit.**

**Preserve unexpected results.**

**Training loss is not the final objective.**

**Use abstractions after understanding what they abstract.**