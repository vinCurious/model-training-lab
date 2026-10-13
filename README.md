# Model Training Lab

> From first principles to modern LLM training and fine-tuning.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-000000?logo=apple&logoColor=white)
![Progress](https://img.shields.io/badge/Current-Module%2003-2ea44f)

A hands-on study of how machine-learning models are trained, starting with gradients and optimization and progressing toward language models, Transformers, fine-tuning, parameter-efficient training, quantization, and evaluation.

The emphasis is on understanding the mechanics behind the abstractions:

> **Build it, inspect it, break it, measure it, then abstract it.**

---

## Roadmap

| Module | Focus | Status |
|:---:|---|:---:|
| `00` | Environment, PyTorch, MPS & reproducibility | ✅ |
| `01` | Training fundamentals from first principles | ✅ |
| `02` | Neural networks & deep-network trainability | ✅ |
| `03` | Language modeling & next-token prediction | ⏳ Current |
| `04` | Transformer / GPT from scratch | Planned |
| `05` | Pretrained models & full fine-tuning | Planned |
| `06` | Supervised fine-tuning (SFT) | Planned |
| `07` | LoRA & parameter-efficient fine-tuning | Planned |
| `08` | Quantization, QLoRA & MLX | Planned |
| `09` | Evaluation & error analysis | Planned |
| `10` | End-to-end model-training capstone | Planned |

Search, retrieval, ranking, and RAG experiments are intentionally kept outside this repository so this project remains focused on **model training and post-training**.

---

## Completed Work

### Module 01 — Training Fundamentals

Built the training loop from first principles and explored:

- tensors, parameters, and computational graphs
- manual gradient derivation vs PyTorch autograd
- gradient descent and learning-rate failure modes
- gradient accumulation
- SGD, momentum, Adam, and AdamW
- train/validation splits
- overfitting, model capacity, early stopping, and weight decay

The experiments move from a manually trained linear model to PyTorch abstractions while keeping the underlying mechanics explicit.

### Module 02 — Neural Networks

Extended the same ideas to deeper neural networks:

- linear models vs nonlinear function approximation
- ReLU and learned hidden representations
- tensor shapes through multilayer networks
- vanishing and exploding activations/gradients
- He/Kaiming and Xavier/Glorot initialization
- `tanh` saturation
- LayerNorm
- residual connections
- PreNorm residual blocks
- gradient-flow diagnostics during training

Several experiments intentionally use poor initialization or architecture choices to make failure modes observable before introducing techniques that address them.

---

## Experimental Approach

Experiments generally follow:

```text
question
→ hypothesis
→ minimal experiment
→ measure
→ visualize
→ interpret
→ modify one assumption
→ repeat
```

Training success is not judged by loss alone. Depending on the experiment, the repository examines:

- training and validation loss
- tensor shapes
- activation statistics
- gradient magnitudes
- parameter counts
- generalization behavior
- model outputs
- memory and compute requirements

Notebook outputs are intentionally retained when they form part of the experimental result.

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
├── data/
├── outputs/
└── checkpoints/
```

Notebooks contain the experiment narrative, code, plots, and measured results.

Each module's `notes.md` serves as a compact reference for the concepts introduced in that module.

As the project progresses, reusable training and evaluation logic will move from notebooks into Python modules and scripts where appropriate.

---

## Environment

Current experiments are designed to run locally on Apple Silicon using:

- Python 3.12
- PyTorch
- Apple MPS
- Jupyter
- matplotlib

The project starts with local compute and introduces larger models or additional infrastructure only when required by the experiment.

---

## Next

**Module 03 — Language Modeling**

The next stage moves from scalar and vector prediction to sequences:

```text
text
→ tokens
→ embeddings
→ context
→ next-token logits
→ cross-entropy loss
→ generation
```

The goal is to understand next-token prediction from first principles before building a Transformer.

---

## Principles

**Understand before abstracting.**

**Measure rather than assume.**

**Break things intentionally.**

**Keep tensor shapes explicit.**

**Training loss is not the final objective.**