# Model Training Lab

> **From first principles to modern LLM fine-tuning.**

A hands-on lab for understanding how models actually learn — starting with gradients and optimization, then progressing through neural networks, language modeling, Transformers, fine-tuning, LoRA, quantization, and evaluation.

The goal is to **understand before abstracting**: implement core ideas directly, inspect what happens during training, deliberately break things, and only then introduce higher-level frameworks.

---

## 🧭 Roadmap

| Module | Focus | Status |
|---|---|---|
| `00` | Environment, PyTorch, Apple MPS, reproducibility | ✅ Complete |
| `01` | Training fundamentals from first principles | 🚧 In progress |
| `02` | Neural networks, optimization & generalization | ⏳ Planned |
| `03` | Language modeling & next-token prediction | ⏳ Planned |
| `04` | Transformer / GPT from scratch | ⏳ Planned |
| `05` | Pretrained models & full fine-tuning | ⏳ Planned |
| `06` | Supervised fine-tuning (SFT) | ⏳ Planned |
| `07` | LoRA & parameter-efficient fine-tuning | ⏳ Planned |
| `08` | Quantization, QLoRA & MLX | ⏳ Planned |
| `09` | Evaluation & error analysis | ⏳ Planned |
| `10` | LLM + retrieval/search capstone | ⏳ Planned |

The roadmap may evolve as experiments reveal useful directions to investigate.

---

## 🔬 Learning Approach

```text
Question
   ↓
Hypothesis
   ↓
Minimal experiment
   ↓
Measure
   ↓
Visualize
   ↓
Explain
   ↓
Change one variable
   ↓
Repeat
```

Each experiment is designed around a concrete question rather than simply running a training script.

Examples:

- What does a gradient actually tell us?
- Why does learning rate affect convergence?
- Why can validation loss increase while training loss decreases?
- What happens when gradients are not reset?
- Why does next-token prediction produce a language model?
- Why does causal self-attention require masking?
- What changes during full fine-tuning?
- What exactly changes during LoRA fine-tuning?
- Does a larger LoRA rank always help?
- Why is lower training loss not necessarily a better model?
- How should model quality be evaluated beyond loss?

---

## 🧠 Core Principles

### 1. Understand before abstracting

Start with:

```text
forward
→ loss
→ backward
→ gradients
→ parameter update
```

before relying on:

```python
trainer.train()
```

The objective is to understand what the abstraction is hiding.

---

### 2. Inspect what the model is doing

Training experiments should expose things like:

```text
tensor shapes
parameter values
gradients
loss curves
validation behavior
model outputs
memory usage
parameter counts
```

A training pipeline should not feel like a black box.

---

### 3. Break things intentionally

Useful experiments include:

```text
learning rate too high
learning rate too low
missing gradient reset
tiny dataset overfitting
frozen parameters
different model capacity
different batch sizes
different dataset sizes
different LoRA ranks
```

Failure modes are part of the curriculum.

---

### 4. Measure, don't assume

Training loss alone is not enough.

As the project progresses, experiments will include:

```text
training loss
validation loss
task metrics
generation quality
error analysis
memory usage
latency
parameter counts
held-out evaluation sets
```

The goal is to understand whether a model actually improved, not simply whether optimization succeeded.

---

### 5. Keep tensor shapes explicit

Examples:

```text
Linear regression:
[examples, features]

Neural network:
[batch, features]

Language model:
[batch, sequence]

Transformer hidden state:
[batch, sequence, hidden_dim]

Attention:
[batch, heads, sequence, sequence]
```

Shape reasoning becomes increasingly important as models get more complex.

---

### 6. Prefer reproducible experiments

Important experiments should be recoverable from:

```text
code
configuration
random seeds
dataset version
model version
metrics
documented assumptions
```

---

## ⚙️ Current Environment

Initial development environment:

```text
Apple Silicon M3
16 GB unified memory
Python 3.12
PyTorch
Apple MPS
VS Code
Jupyter
Git
```

Verified accelerator path:

```text
PyTorch
   ↓
MPS backend
   ↓
Apple GPU
   ↓
tensor operations on mps:0
```

The project intentionally starts locally before introducing cloud GPUs.

---

## 🛠️ Tooling Progression

Tools are introduced only when there is a concrete reason to use them.

```text
Python / NumPy
        ↓
PyTorch
        ↓
Jupyter + matplotlib
        ↓
Hugging Face Datasets
        ↓
Transformers
        ↓
PEFT / LoRA
        ↓
TRL
        ↓
MLX / MLX-LM
        ↓
Experiment tracking
        ↓
LLM evaluation tooling
```

### Core training

- Python
- NumPy
- PyTorch
- Jupyter
- matplotlib

### Models and datasets

- Hugging Face Hub
- Transformers
- Datasets
- Tokenizers

### LLM post-training

- PEFT
- LoRA
- TRL
- quantization tooling

### Apple Silicon

- MLX
- MLX-LM

### Experiment tracking

- TensorBoard
- MLflow
- Weights & Biases

### Evaluation

- held-out evaluation sets
- task-specific metrics
- standardized language-model evaluation tooling
- error analysis

### Engineering workflow

- Git
- GitHub
- virtual environments
- structured configs
- checkpoints
- artifacts
- reproducible runs

Public datasets and widely used open-source models will be preferred where practical.

---

## 📁 Repository Structure

Current structure:

```text
model-training-lab/
│
├── README.md
├── requirements.txt
│
├── module00_setup/
│   ├── system_check.py
│   └── notes.md
│
├── module01_training_fundamentals/
│   ├── 01_manual_linear_regression.ipynb
│   └── notes.md
│
├── data/
├── outputs/
└── checkpoints/
```

As experiments become more realistic, the repository will evolve toward:

```text
configs/
src/
scripts/
notebooks/
evals/
```

Notebooks are primarily used for:

```text
exploration
visualization
interactive analysis
small controlled experiments
```

Reusable training, evaluation, and data-processing logic will gradually move into Python modules and scripts.

---

## 🧩 Module 00 — Environment & Reproducibility

Goal:

```text
Python
→ virtual environment
→ PyTorch
→ MPS
→ Apple GPU
```

The environment includes a reproducible system check that verifies tensor allocation and matrix multiplication on the Apple GPU.

Example compute path:

```text
Python 3.12
   ↓
project .venv
   ↓
PyTorch
   ↓
MPS
   ↓
mps:0
```

This creates a clean baseline before training experiments begin.

---

## 🚧 Module 01 — Training From First Principles

Starting problem:

\[
y = 3x + 2 + \epsilon
\]

Model:

\[
\hat{y} = wx + b
\]

Objective:

```text
prediction
   ↓
loss
   ↓
gradient
   ↓
parameter update
   ↓
better prediction
```

The first implementation intentionally avoids high-level training abstractions.

Topics include:

```text
features and targets
parameters vs hyperparameters
mean squared error
computational graphs
automatic differentiation
gradients
gradient descent
learning rate
gradient accumulation
optimization stability
train vs validation behavior
generalization
```

The goal is to observe directly how parameters move toward values that reduce loss.

---

## 📉 Optimization Experiments

A major focus of the early modules is understanding optimization behavior experimentally.

Examples:

```text
reasonable learning rate
→ stable convergence

very small learning rate
→ slow progress

very large learning rate
→ overshooting / instability

gradients not cleared
→ unintended accumulation
```

Later experiments will compare:

```text
manual gradient descent
SGD
Momentum
Adam
AdamW
learning-rate schedules
weight decay
```

---

## 🧮 From a Tiny Model to an LLM

The same basic training loop appears throughout the project.

At the beginning:

```text
x
↓
wx + b
↓
loss
↓
backpropagation
↓
parameter update
```

Later:

```text
tokens
↓
embeddings
↓
Transformer blocks
↓
next-token logits
↓
cross-entropy loss
↓
backpropagation
↓
parameter update
```

The scale becomes dramatically larger, but the underlying optimization structure remains recognizable.

At a high level, training can be viewed as:

\[
\theta^*
=
\arg\min_{\theta}
\frac{1}{N}
\sum_{i=1}^{N}
L(f_{\theta}(x_i), y_i)
\]

where:

- \(\theta\) represents model parameters
- \(f_{\theta}\) is the model
- \(x_i\) is an input
- \(y_i\) is a target
- \(L\) is the loss function

Understanding this continuity is one of the main goals of the lab.

---

## 🔤 Language Modeling Path

The language-modeling portion will progress through:

```text
text
   ↓
tokenization
   ↓
token IDs
   ↓
embeddings
   ↓
context representation
   ↓
next-token logits
   ↓
softmax distribution
   ↓
next-token loss
```

Topics will include:

- vocabulary
- tokenization
- embeddings
- context length
- logits
- softmax
- cross-entropy
- autoregressive modeling
- sampling
- temperature
- generation

---

## 🧱 Transformer From Scratch

The project will eventually build a small Transformer directly.

Conceptual structure:

```text
token embeddings
        +
position information
        ↓
causal self-attention
        ↓
residual connection
        ↓
normalization
        ↓
MLP
        ↓
residual connection
        ↓
language-model head
```

Tensor shapes will be tracked explicitly.

Example:

```text
tokens:
[B, T]

embeddings:
[B, T, C]

Q / K / V:
[B, H, T, D]

attention:
[B, H, T, T]

logits:
[B, T, vocabulary_size]
```

Where:

```text
B = batch size
T = sequence length
C = hidden dimension
H = attention heads
D = head dimension
```

---

## 🔧 Fine-Tuning Path

The fine-tuning portion will compare several approaches.

### Full fine-tuning

```text
pretrained model
      ↓
training data
      ↓
update most/all model parameters
```

### LoRA

```text
frozen pretrained weights
        +
small trainable low-rank adapters
```

Conceptually:

\[
W' = W + BA
\]

where the base weight \(W\) remains frozen and only the low-rank update is trained.

Experiments will investigate:

```text
LoRA rank
LoRA alpha
target modules
learning rate
dataset size
trainable parameter count
model quality
```

### Quantization / QLoRA

```text
quantized base model
        +
trainable LoRA adapters
```

This allows larger models to be explored within limited hardware memory.

---

## 🍎 Apple Silicon Experiments

Later modules will compare PyTorch/MPS workflows with Apple-native ML tooling.

Planned tools:

```text
PyTorch + MPS
vs
MLX
vs
MLX-LM
```

Topics will include:

```text
unified memory
model size
quantization
training memory
inference memory
batch size
context length
adapter training
```

---

## 📊 Evaluation

Model training and model quality are not the same thing.

The evaluation portion will investigate:

```text
training loss
validation loss
held-out task performance
generation quality
domain accuracy
error categories
hallucination behavior
generalization
catastrophic forgetting
```

The goal is to avoid the assumption:

```text
lower training loss
=
better model
```

Evaluation should answer whether the model improved on the behavior we actually care about.

---

## 🔭 Long-Term Direction

The overall progression is:

```text
training fundamentals
        ↓
neural networks
        ↓
optimization
        ↓
language modeling
        ↓
self-attention
        ↓
Transformer from scratch
        ↓
pretrained models
        ↓
full fine-tuning
        ↓
supervised fine-tuning
        ↓
LoRA / PEFT
        ↓
quantization / QLoRA
        ↓
evaluation
        ↓
LLM + retrieval/search systems
```

---

## 🧪 Capstone Direction

The final project is expected to combine model training with areas such as:

```text
retrieval
search
embeddings
ranking
vector search
recommendation systems
RAG
evaluation
ML systems
```

Potential experiment structure:

```text
Base model
   ↓
domain dataset
   ↓
SFT
   ↓
LoRA
   ↓
quantized LoRA
   ↓
held-out evaluation
   ↓
error analysis
```

The goal is to compare approaches rather than simply produce one fine-tuned checkpoint.

---

## 📌 Current Focus

> **Module 01 — understanding model training from first principles.**

Current progression:

```text
✅ environment setup
✅ PyTorch + MPS verification
✅ Git repository setup
🚧 manual linear regression
⏳ autograd exploration
⏳ learning-rate experiments
⏳ PyTorch nn.Module
⏳ optimizer comparison
```

---

## Philosophy

> **Understand before abstracting.**

> **Measure rather than assume.**

> **Break things intentionally.**

> **Training loss is not the final objective.**

> **Build intuition from experiments, not just APIs.**