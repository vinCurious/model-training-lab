# Model Training Lab

> **From first principles to modern LLM fine-tuning.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-000000?logo=apple&logoColor=white)
![Progress](https://img.shields.io/badge/Current-Module%2001-2ea44f)
![Approach](https://img.shields.io/badge/Approach-First%20Principles-6f42c1)

A hands-on model-training lab for understanding how models actually learn.

The progression starts with **tensors, loss functions, gradients, optimization, and generalization**, then moves toward language modeling, Transformers, full fine-tuning, supervised fine-tuning, LoRA, quantization, and rigorous LLM evaluation.

The guiding principle is simple:

> **Understand before abstracting.**

Rather than starting with high-level training APIs, each important concept is first implemented, inspected, measured, and deliberately stressed before introducing the abstraction that automates it.

---

## 🧭 Roadmap

```text
Training fundamentals
        ↓
Neural networks & optimization
        ↓
Language modeling
        ↓
Self-attention
        ↓
Transformer / GPT from scratch
        ↓
Pretrained models
        ↓
Full fine-tuning
        ↓
Supervised fine-tuning
        ↓
LoRA / PEFT
        ↓
Quantization / QLoRA
        ↓
Evaluation
        ↓
LLM + retrieval/search systems
```

| Module | Focus | Status |
|:---:|---|:---:|
| `00` | Environment, PyTorch, Apple MPS & reproducibility | ✅ Complete |
| `01` | Training fundamentals from first principles | ✅ Complete |
| `02` | Neural networks, optimization & generalization | ⏭️ Next / Planned |
| `03` | Language modeling & next-token prediction | ⏳ Planned |
| `04` | Transformer / GPT from scratch | ⏳ Planned |
| `05` | Pretrained models & full fine-tuning | ⏳ Planned |
| `06` | Supervised fine-tuning (SFT) | ⏳ Planned |
| `07` | LoRA & parameter-efficient fine-tuning | ⏳ Planned |
| `08` | Quantization, QLoRA & MLX | ⏳ Planned |
| `09` | Evaluation & error analysis | ⏳ Planned |
| `10` | LLM + retrieval/search capstone | ⏳ Planned |

The roadmap will evolve as experiments expose useful questions worth investigating further.

---

## 🔬 Experiment-Driven Approach

Each meaningful experiment follows roughly the same loop:

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
Break or modify one assumption
   ↓
Repeat
```

The goal is not simply to make training succeed.

The goal is to understand **why it succeeds, why it fails, and what the model is actually doing**.

Questions explored throughout the project include:

- What does a gradient actually tell us?
- Why does gradient descent reduce loss?
- What happens when the learning rate is too large?
- What happens when it is too small?
- Why can training loss fall while validation loss rises?
- What happens if gradients are not cleared?
- What changes when parameters are frozen?
- Why does next-token prediction produce a language model?
- Why does causal self-attention require masking?
- What changes during full fine-tuning?
- How can LoRA modify model behavior while training only a small fraction of parameters?
- Does a larger LoRA rank always improve results?
- Why can a lower training loss still produce a worse language model?
- How should model quality be evaluated beyond loss?

---

## 🧠 Core Principles

### 1. Understand before abstracting

Start by understanding:

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

The abstraction is useful only after understanding what it hides.

---

### 2. Inspect what the model is doing

Training experiments should expose things such as:

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

Failure modes are part of the curriculum.

Experiments will deliberately include:

```text
learning rate too high
learning rate too low
gradients not cleared
tiny dataset overfitting
frozen parameters
different batch sizes
different model capacities
different dataset sizes
different LoRA ranks
```

A successful default configuration often teaches less than a controlled failure.

---

### 4. Measure rather than assume

Training success should be supported by evidence.

As the project progresses, experiments will examine:

```text
training loss
validation loss
task metrics
parameter changes
gradient behavior
generation quality
error categories
memory usage
latency
held-out evaluation sets
```

---

### 5. Keep tensor shapes explicit

A tensor such as:

```text
[8, 12, 512, 512]
```

is not useful until the dimensions have meaning:

```text
batch
× attention heads
× query positions
× key positions
```

Tensor-shape reasoning is treated as a core ML engineering skill throughout the project.

---

### 6. Prefer reproducible experiments

Important experiments should be recoverable from:

```text
code
configuration
random seeds
dataset versions
model versions
metrics
documented assumptions
```

---

### 7. Training loss is not the final objective

The eventual question is not:

> Did optimization reduce the training loss?

It is:

> Did the model improve on the behavior we actually care about?

That distinction becomes increasingly important during fine-tuning and LLM evaluation.

---

## 🚧 Current Focus

### Module 01 — Training From First Principles

The first training problem is intentionally small:

$$
y = 3x + 2 + \epsilon
$$

with a model:

$$
\hat{y} = wx + b
$$

The objective is to observe the entire learning process directly:

```text
data
  ↓
forward pass
  ↓
prediction
  ↓
loss
  ↓
gradient
  ↓
parameter update
  ↓
repeat
```

Module 1 completed:

- [x] Python environment
- [x] PyTorch installation
- [x] Apple MPS verification
- [x] Git/GitHub setup
- [x] Tensor fundamentals
- [x] Manual linear regression
- [x] Computational graphs
- [x] Manual gradient derivation
- [x] Autograd verification
- [x] Gradient descent
- [x] Learning-rate experiments
- [x] Gradient accumulation
- [x] `nn.Module`
- [x] SGD
- [x] Momentum
- [x] Adam / AdamW
- [x] Train / validation split
- [x] Overfitting & generalization

---

## 🧩 Module 00 — Environment & Reproducibility

The initial setup establishes a reproducible local training environment:

```text
Python
   ↓
virtual environment
   ↓
PyTorch
   ↓
MPS
   ↓
Apple GPU
```

The system check verifies:

- Python runtime
- PyTorch version
- MPS availability
- accelerator selection
- tensor allocation
- matrix multiplication on the accelerator

Verified compute path:

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

## 📐 Tensor Fundamentals

A tensor is an n-dimensional numerical array.

A useful mental model:

```text
Tensor
├── values
├── shape
├── dtype
├── device
└── autograd metadata
```

Important properties:

```python
tensor.shape
tensor.ndim
tensor.dtype
tensor.device
tensor.requires_grad
```

When inspecting a tensor, ask:

```text
1. What does it represent?
2. What does each dimension mean?
3. What is its shape?
4. What dtype is it?
5. Which device is it on?
6. Does it require gradients?
```

Example:

```text
x
→ input data
→ [100 examples, 1 feature]
→ float32
→ CPU initially
→ gradients not required

w
→ learnable parameter
→ [1, 1]
→ float32
→ CPU initially
→ gradients required
```

A useful debugging rule:

> **Never look at a tensor shape without also asking what each dimension represents.**

Detailed tensor notes live inside the Module 01 documentation.

---

## 🧮 From a Tiny Model to an LLM

The same fundamental training structure appears throughout the project.

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

The models become dramatically larger, but the underlying optimization problem remains recognizable.

At a high level:

$$
\theta^*
=
\arg\min_{\theta}
\frac{1}{N}
\sum_{i=1}^{N}
L(f_{\theta}(x_i), y_i)
$$

where:

- $\theta$ represents model parameters
- $f_{\theta}$ is the model
- $x_i$ is an input
- $y_i$ is a target
- $L$ is the loss function

Understanding this continuity is one of the central goals of the project.

---

## 📉 Optimization Experiments

Early modules explore optimization directly.

Examples:

```text
reasonable learning rate
→ stable convergence

very small learning rate
→ correct direction, very slow progress

very large learning rate
→ overshooting / instability / divergence

gradients not cleared
→ unintended accumulation
```

Later comparisons will include:

```text
manual gradient descent
        ↓
SGD
        ↓
Momentum
        ↓
Adam
        ↓
AdamW
        ↓
learning-rate schedules
```

Related topics include:

- initialization
- gradient magnitude
- weight decay
- regularization
- batch size
- gradient accumulation
- optimization stability
- convergence

---

## 🧠 Neural Networks

After basic optimization, the project moves from:

```text
single linear model
```

to:

```text
linear layers
     ↓
nonlinear activations
     ↓
multiple layers
     ↓
learned representations
```

Topics include:

- affine transformations
- activations
- nonlinear decision boundaries
- initialization
- backpropagation through multiple layers
- vanishing and exploding gradients
- normalization
- regularization
- capacity
- bias vs variance
- train / validation behavior

---

## 🔤 Language Modeling

The language-modeling path introduces next-token prediction:

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
probability distribution
   ↓
next-token loss
```

Topics include:

- vocabulary
- tokenization
- embeddings
- context length
- logits
- softmax
- cross-entropy
- maximum likelihood
- autoregressive modeling
- sampling
- temperature
- generation

The objective is to connect ordinary classification concepts directly to language-model training.

---

## 🧱 Transformer From Scratch

A small Transformer will be implemented directly before relying completely on high-level libraries.

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
tokens
[B, T]

embeddings
[B, T, C]

Q / K / V
[B, H, T, D]

attention
[B, H, T, T]

logits
[B, T, vocabulary_size]
```

where:

```text
B = batch size
T = sequence length
C = hidden dimension
H = attention heads
D = head dimension
```

---

## 🔧 Fine-Tuning Path

### Full Fine-Tuning

```text
pretrained model
      ↓
task/domain data
      ↓
update most or all parameters
```

This establishes the baseline for understanding parameter-efficient approaches.

---

### Supervised Fine-Tuning

Instruction-style datasets will introduce:

```text
system / user / assistant messages
            ↓
chat template
            ↓
tokenization
            ↓
labels
            ↓
causal language-model loss
```

Topics include:

- instruction data
- chat templates
- masking
- causal LM loss
- checkpointing
- validation
- model behavior before vs after fine-tuning

---

### LoRA / PEFT

Rather than modifying the entire pretrained weight matrix:

$$
W
$$

LoRA models an update approximately as:

$$
W' = W + BA
$$

where the base weights can remain frozen and the low-rank matrices are trained.

Experiments will investigate:

```text
LoRA rank
LoRA alpha
target modules
dropout
learning rate
dataset size
trainable parameter count
model quality
```

Questions include:

> How little trainable capacity is actually needed?

and:

> Does increasing LoRA rank always improve performance?

---

## 📦 Quantization & QLoRA

Quantization experiments will explore lower-precision model representations such as:

```text
FP32
FP16
BF16
INT8
INT4
```

Questions include:

- How much memory is saved?
- What accuracy tradeoffs appear?
- Why can quantized models still be fine-tuned using adapters?
- How does quantization change local model feasibility?

Conceptually:

```text
quantized base model
        +
trainable LoRA adapters
```

---

## 🍎 Apple Silicon Experiments

Local experiments will compare approaches such as:

```text
PyTorch + MPS
       vs
MLX
       vs
MLX-LM
```

Topics include:

- unified memory
- model size
- training memory
- inference memory
- quantization
- context length
- batch size
- adapter training
- throughput

The aim is not just to make models fit, but to understand **why they fit**.

---

## 📊 Evaluation

Model training and model quality are different problems.

Evaluation work will include:

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
latency
memory usage
```

A central question is:

```text
lower training loss
        ≠
automatically better model
```

Evaluation should determine whether the model improved on the behavior that actually matters.

---

## 🛠️ Tooling Progression

Tools are introduced when they solve a concrete problem rather than installed up front.

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

### Models & datasets

Planned:

- Hugging Face Hub
- Transformers
- Datasets
- Tokenizers

### LLM post-training

Planned:

- PEFT
- LoRA
- TRL
- quantization tooling

### Apple Silicon

Planned:

- MLX
- MLX-LM

### Experiment tracking

Planned exposure:

- TensorBoard
- MLflow
- Weights & Biases

### Evaluation

Planned:

- task-specific metrics
- held-out evaluation sets
- standardized language-model evaluation tooling
- structured error analysis

### Engineering workflow

- Git
- GitHub
- virtual environments
- reproducible experiments
- structured configs
- checkpoints
- artifacts

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

As the experiments become more realistic, the project will evolve toward:

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
controlled experiments
```

Reusable logic will gradually move into Python modules and scripts.

---

## 🗂️ Notebook vs Production Code

Early experiments intentionally use notebooks because they make it easy to:

```text
run one step
inspect a tensor
change a variable
plot the result
form a new hypothesis
rerun
```

As concepts become stable:

```text
notebook exploration
        ↓
reusable Python code
        ↓
config-driven experiments
        ↓
reproducible training pipeline
```

This mirrors the progression from research exploration toward engineering-quality training workflows.

---

## 🌐 Public Datasets & Models

Where useful, experiments will use public datasets and open models rather than only synthetic examples.

The progression is expected to include:

```text
synthetic data
      ↓
small public datasets
      ↓
language-model datasets
      ↓
instruction datasets
      ↓
domain-specific datasets
      ↓
held-out evaluation sets
```

Dataset selection will consider:

- task relevance
- quality
- license
- size
- reproducibility
- train / validation / test separation

---

## 🔭 Capstone Direction

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

A potential experiment structure:

```text
Base model
   ↓
domain dataset
   ↓
baseline evaluation
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
   ↓
system-level conclusions
```

The objective will be to **compare approaches**, not simply produce one fine-tuned checkpoint.

---

## 🧪 Experiment Documentation

Meaningful experiments should capture:

```text
Question
Hypothesis
Setup
Variables
Metrics
Results
Plots
Failure modes
Conclusion
Next question
```

This repository is intended to document the reasoning behind experiments, not only their final code.

---

## ⚙️ Current Environment

```text
Apple Silicon M3
16 GB unified memory
Python 3.12
PyTorch
Apple MPS
VS Code
Jupyter
Git
GitHub
```

The project begins with local compute and will introduce other environments only when they serve a specific learning objective.

---

## 📌 Current Status

```text
✅ environment setup
✅ isolated Python environment
✅ PyTorch installation
✅ Apple MPS verification
✅ Git repository setup
✅ public GitHub repository
✅ tensor fundamentals
✅ manual linear regression
✅ autograd exploration
✅ gradient verification
✅ learning-rate experiments
✅ optimizer comparisons
✅ validation, overfitting & regularization
⏭️ Module 02 — Next / Planned
```

---

## Philosophy

> **Understand before abstracting.**

> **Measure rather than assume.**

> **Break things intentionally.**

> **Make tensor shapes explicit.**

> **Training loss is not the final objective.**

> **Build intuition from experiments, not just APIs.**