# Module 01 — Training Fundamentals

This file is the durable reference for Module 1.

The notebooks contain the executable experiments, plots, measured results, and detailed interpretations:

- [Manual linear regression](01_manual_linear_regression.ipynb) — tensors, gradients, updates, optimizer behavior, and controlled failure cases.
- [Validation and overfitting](02_train_validation_overfitting.ipynb) — generalization, checkpoints, regularization, early stopping, and model capacity.

---

## What This Module Establishes

The core training loop is:

```text
data
↓
model
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

Module 1 builds the intuition behind:

```text
tensor
→ parameter
→ prediction
→ loss
→ gradient
→ optimizer
→ training
→ validation
→ generalization
→ overfitting
→ regularization
```

The guiding principle is:

> **Understand the mechanics before relying on abstractions.**

---

# Tensor Fundamentals

A tensor is an n-dimensional numerical array.

A useful mental model is:

```text
Tensor
├── values
├── shape
├── dtype
├── device
└── autograd metadata
```

PyTorch tensors support:

- vectorized numerical operations
- matrix multiplication
- broadcasting
- accelerator execution
- multiple numeric precisions
- automatic differentiation

---

## Common Tensor Creation

### `torch.randn`

Creates values sampled from a standard normal distribution.

```python
w = torch.randn(1, 1)
```

Conceptually:

```text
rand → random
n    → normal distribution
```

---

### `torch.zeros`

Creates a tensor filled with zeros.

```python
b = torch.zeros(1)
```

Initialization and gradient tracking are separate concepts:

```python
b = torch.zeros(
    1,
    requires_grad=True
)
```

means:

```text
initial value = 0
gradient tracking = enabled
```

---

## Important Tensor Properties

| Property | Meaning | Example |
|---|---|---|
| `shape` | Size of each axis | `[100, 1]` |
| `ndim` | Number of axes | Matrix → `2` |
| `dtype` | Numeric representation | `torch.float32` |
| `device` | Where values live | `cpu`, `mps`, `cuda` |
| `requires_grad` | Whether autograd tracks derivatives | Usually `True` for learnable parameters |

A strong debugging rule is:

> **Never inspect a tensor shape without asking what every dimension means.**

Example:

```text
[100, 1]
→ 100 examples
→ 1 feature per example
```

Later:

```text
[8, 512, 768]
→ batch
→ token positions
→ hidden features
```

---

## Shape Reasoning

For linear regression:

| Tensor | Role | Shape | Gradient role |
|---|---|---|---|
| `x` | Input data | `[N, 1]` | Fixed data |
| `y` | Targets | `[N, 1]` | Fixed data |
| `w` | Weight | `[1, 1]` | Learnable parameter |
| `b` | Bias | `[1]` | Learnable parameter |
| `y_pred` | Predictions | `[N, 1]` | Derived through graph |
| `loss` | Objective | `[]` | Scalar derived through graph |

For:

```python
y_pred = x @ w + b
```

the multiplication is:

```text
[N, 1] @ [1, 1]
→ [N, 1]
```

and the bias is broadcast across rows.

Broadcasting is useful but can hide bugs.

For example:

```text
[N, 1] - [N]
```

can produce:

```text
[N, N]
```

instead of one error per example.

---

## Matrix Multiplication vs Elementwise Multiplication

Matrix multiplication:

```python
x @ w
```

Elementwise multiplication:

```python
x * w
```

For matrix multiplication, inner dimensions must match:

```text
[32, 128] @ [128, 256]
→ [32, 256]
```

---

## dtype

Common dtypes:

```text
float32
float16
bfloat16
float64
int64
bool
```

Typical uses:

```text
float32
→ standard training precision

float16 / bfloat16
→ reduced precision
→ lower memory
→ faster computation on supported hardware

int64
→ token IDs
→ class IDs

bool
→ masks
```

Precision affects:

```text
memory
speed
numerical stability
hardware compatibility
```

---

## Device

Possible devices include:

```text
cpu
mps
cuda
```

Move a tensor with:

```python
x = x.to(device)
```

Tensors participating in one computation generally need compatible dtypes and the same device.

---

# Parameters and Hyperparameters

A **parameter** is learned from data.

Examples:

```text
weights
biases
```

A **hyperparameter** is chosen by us.

Examples:

```text
learning rate
epochs
momentum
weight decay
batch size
model size
early-stopping patience
```

Useful distinction:

```text
parameter
→ learned during training

hyperparameter
→ configures training
```

---

# `requires_grad`

Example:

```python
w = torch.randn(
    1,
    1,
    requires_grad=True
)
```

`requires_grad=True` tells PyTorch:

> Track operations involving this tensor so derivatives can later be computed with respect to it.

Typical setup:

```text
x.requires_grad = False
y.requires_grad = False

w.requires_grad = True
b.requires_grad = True
```

Important distinction:

```text
requires_grad=True
→ track operations

loss.backward()
→ compute gradients

optimizer.step()
→ modify parameters
```

---

# Computational Graph and Autograd

Suppose:

```python
y_pred = x @ w + b
loss = ((y_pred - y) ** 2).mean()
```

PyTorch tracks the dependency structure:

```text
w ──────┐
        ├──> y_pred ───> loss
b ──────┘
```

Then:

```python
loss.backward()
```

uses the chain rule to compute:

$$
\frac{\partial L}{\partial w}
$$

and:

$$
\frac{\partial L}{\partial b}
$$

and stores them in:

```python
w.grad
b.grad
```

Parameters such as `w` and `b` are typically leaf tensors.

Derived tensors such as `y_pred` and `loss` participate in the graph but are not themselves model parameters.

---

# Linear Regression Model

The synthetic relationship is approximately:

$$
y = 3x + 2 + \epsilon
$$

The model is:

$$
\hat{y} = wx + b
$$

where:

```text
w → weight / slope
b → bias / intercept
```

The weight controls the tilt of the line.

The bias controls its vertical position.

When:

$$
x = 0
$$

then:

$$
\hat{y} = b
$$

---

# Loss

The regression experiments use Mean Squared Error:

$$
L
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
$$

Squaring errors:

```text
prevents positive and negative errors from canceling
+
penalizes larger errors more strongly
```

The loss is a scalar:

```text
shape = []
```

---

# Manual Gradients

For:

$$
\hat{y}_i = x_iw+b
$$

and:

$$
L
=
\frac{1}{N}
\sum_i
(\hat{y}_i-y_i)^2
$$

the gradients are:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{N}
\sum_i
(\hat{y}_i-y_i)x_i
$$

and:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{N}
\sum_i
(\hat{y}_i-y_i)
$$

The notebook verifies that the manual gradients match PyTorch autograd to floating-point precision.

Autograd automates the calculus; it does not change the underlying mathematics.

---

# What a Gradient Means

A gradient is a local sensitivity.

It answers:

> If this parameter changes slightly, how does the loss change?

If:

```text
gradient < 0
```

increasing the parameter locally reduces the loss.

If:

```text
gradient > 0
```

decreasing the parameter locally reduces the loss.

The gradient does not directly tell us the final correct parameter value.

---

# Gradient Descent

Gradient descent updates parameters using:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta L
$$

where:

```text
θ → parameter
η → learning rate
∇L → gradient
```

Conceptually:

```text
gradient
→ direction + local steepness

learning rate
→ step size
```

---

# Learning Rate

Typical behavior:

```text
too small
→ slow convergence

reasonable
→ stable convergence

too large
→ overshooting
→ divergence
```

The learning rate is not a property of the model alone.

It interacts with:

```text
optimizer
momentum
model architecture
gradient scale
batch size
```

---

# Canonical Training Loop

```python
for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)

    loss.backward()
    optimizer.step()
```

Interpretation:

```text
optimizer.zero_grad()
→ clear previous gradients

model(x_train)
→ forward pass

loss_fn(...)
→ compute objective

loss.backward()
→ compute gradients

optimizer.step()
→ update parameters
```

---

# Epochs and Batches

An **epoch** is one complete pass through the training dataset.

The early Module 1 experiments use full batches:

```text
all training examples
→ one forward pass
→ one backward pass
→ one parameter update
```

Later, with mini-batches:

```text
batch 1 → update
batch 2 → update
batch 3 → update
...
↓
all examples seen once
↓
1 epoch
```

---

# Gradient Clearing and Accumulation

PyTorch accumulates gradients by default.

Without clearing:

```text
backward #1 → g1
backward #2 → g1 + g2
backward #3 → g1 + g2 + g3
```

For ordinary training:

```python
optimizer.zero_grad()
```

clears old gradients before the next backward pass.

---

## Intentional Gradient Accumulation

Gradient accumulation can also be deliberate.

Example:

```text
microbatch 1
microbatch 2
microbatch 3
microbatch 4
↓
one optimizer update
```

This is useful when memory limits prevent using the desired batch size directly.

Important distinction:

```text
unintentional accumulation
→ bug

controlled accumulation
→ training technique
```

---

# Manual Updates and `torch.no_grad()`

Manual updates use:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

because the update itself should not become part of the differentiable graph.

After updating parameters, recompute predictions if you want to measure the new loss.

---

# PyTorch Model Abstractions

## `nn.Module`

`nn.Module` organizes:

```text
parameters
layers
forward computation
nested components
```

---

## `nn.Linear`

```python
nn.Linear(in_features, out_features)
```

computes:

$$
y = xW^T+b
$$

and stores weights as:

```text
[out_features, in_features]
```

Always inspect shapes rather than assuming layout.

---

## `nn.MSELoss`

```python
loss_fn = nn.MSELoss()
```

matches the manual mean-squared-error objective used in the notebook.

---

## Optimizers

Manual update:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
```

Optimizer abstraction:

```python
optimizer.step()
```

The optimizer does not compute gradients.

`loss.backward()` computes gradients.

`optimizer.step()` uses them.

---

# Optimizer Concepts

| Optimizer | Main idea | Practical implication |
|---|---|---|
| SGD | Uses current gradient directly | Simple baseline |
| SGD + momentum | Adds decaying velocity | Can accelerate or overshoot |
| Adam | Uses first and second gradient moments | Adaptive per-parameter updates |
| AdamW | Adam-style adaptation + decoupled weight decay | Common in Transformer training |

There is no universal optimizer ranking.

Equal nominal learning rates do not imply equally suitable behavior across optimizers.

---

# Momentum

A simplified form is:

$$
v_t
=
\beta v_{t-1}
+
g_t
$$

followed by:

$$
\theta_{t+1}
=
\theta_t
-
\eta v_t
$$

Momentum adds controlled memory of recent gradients.

It can:

- accelerate persistent directions
- smooth noisy updates
- overshoot minima
- interact strongly with learning rate

Momentum is different from uncleared gradients.

Momentum uses decaying optimizer state; accidental accumulation directly adds gradients into `.grad`.

---

# Adam and AdamW

Adam tracks running estimates of:

```text
first moment
→ gradient direction history

second moment
→ squared gradient magnitude history
```

This produces adaptive parameter updates.

AdamW combines Adam-style adaptation with decoupled weight decay.

AdamW is commonly used in Transformer and LLM training.

---

# Train / Validation Split

Training data is used for:

```text
forward pass
loss
backward pass
parameter updates
```

Validation data is used for:

```text
evaluation
model selection
hyperparameter decisions
```

Validation data should not supply gradients used for training updates.

---

## Splitting Data

The synthetic inputs were shuffled before splitting so both subsets covered the full input range.

Random splitting is not always correct.

Examples:

```text
time-series data
→ preserve time ordering

user-based data
→ often split by user

related documents
→ grouped splitting may prevent leakage
```

The split should reflect the actual deployment problem.

---

# Generalization

Generalization means:

> The model performs usefully on data it did not train on.

Training loss answers:

> How well does the model fit its training examples?

Validation loss answers:

> How well does the learned behavior transfer to unseen examples?

The real objective is generally not:

```text
minimize training loss at any cost
```

but:

```text
learn useful patterns
↓
perform well on unseen data
```

---

# Generalization Gap

The difference between training and validation performance is often called the **generalization gap**.

A gap alone does not prove overfitting.

Look at:

```text
noise
dataset size
sample difficulty
curve behavior over time
```

A classic overfitting signal is:

```text
training loss continues decreasing
while
validation loss stops improving or increases
```

---

# Overfitting

Overfitting occurs when the model increasingly specializes to the training set while unseen performance worsens.

Typical pattern:

```text
early training
→ learn signal

later training
→ fit noise and sample-specific details

training loss
→ decreases

validation loss
→ stops improving or rises
```

---

# Underfitting

Underfitting occurs when the model cannot adequately capture the underlying structure.

Typical pattern:

```text
training loss high
validation loss high
```

Possible causes:

```text
too little capacity
insufficient training
poor optimization
too much regularization
```

---

# Model Capacity

Capacity describes how flexible a model is.

Observed pattern:

```text
too little capacity
→ underfitting

appropriate capacity
→ best generalization

too much capacity
→ stronger overfitting risk
```

The model with the lowest training loss is not necessarily the best model.

---

# Bias–Variance Intuition

Very roughly:

```text
lower-capacity model
→ higher bias
→ lower flexibility

higher-capacity model
→ lower bias
→ potentially higher variance
```

The goal is not to minimize either independently.

The goal is a balance that generalizes well.

---

# Early Stopping

Early stopping monitors validation performance.

```text
train
↓
validate
↓
validation improves?
├── yes → save checkpoint
└── no  → increase patience counter
↓
patience exhausted
↓
stop
↓
restore best checkpoint
```

Important distinction:

```text
last checkpoint
≠
best checkpoint
```

---

## Patience

Patience controls how many non-improving validation steps are allowed before stopping.

Too little patience may stop training too early.

Too much patience may waste compute.

---

## Saving the Best Checkpoint

For an in-memory snapshot:

```python
best_model_state = copy.deepcopy(
    model.state_dict()
)
```

Restore it with:

```python
model.load_state_dict(best_model_state)
```

---

# `model.train()` vs `model.eval()` vs `torch.no_grad()`

These solve different problems.

### `model.train()`

Enables training behavior for mode-dependent layers.

### `model.eval()`

Enables evaluation behavior for mode-dependent layers.

### `torch.no_grad()`

Disables autograd graph construction.

Typical validation:

```python
model.eval()

with torch.no_grad():
    val_pred = model(x_val)
```

`model.eval()` alone does not disable gradients.

---

# Weight Decay

Weight decay discourages large parameter values.

Conceptually:

```text
fit the data
+
discourage unnecessary parameter growth
```

It can reduce aggressive memorization, but it does not automatically improve validation performance.

Regularization strength is another hyperparameter that must be evaluated empirically.

---

# Early Stopping vs Weight Decay

Early stopping:

```text
controls training duration
```

Weight decay:

```text
constrains parameter growth
```

They address overfitting through different mechanisms and can be used together.

---

# Core Experiment Takeaways

## Learning Rate

```text
too small
→ slow

reasonable
→ stable convergence

too large
→ divergence
```

## Gradient Clearing

```text
clear gradients
→ intended update rule

forget to clear
→ accumulated gradients
→ changed optimization behavior
```

## Autograd

```text
manual gradients
≈
PyTorch gradients
```

## PyTorch Abstractions

```text
manual parameters
→ model.parameters()

manual forward equation
→ model(...)

manual MSE
→ nn.MSELoss()

manual update
→ optimizer.step()

manual gradient clearing
→ optimizer.zero_grad()
```

## Optimizers

```text
SGD
→ simple baseline

momentum
→ adds controlled history

Adam
→ adaptive updates

AdamW
→ Adam-style adaptation + decoupled weight decay
```

## Generalization

```text
lowest training loss
≠
best model
```

## Capacity

```text
too small
→ underfit

appropriate
→ best validation

too large
→ overfit
```

---

# Practical Tensor Debugging Checklist

When inspecting a tensor, ask:

1. What does it represent?
2. What does each dimension mean?
3. What is its shape?
4. What dtype is it?
5. Which device is it on?
6. Does it require gradients?

Example:

```text
x
→ input data
→ [100 examples, 1 feature]
→ float32
→ CPU
→ gradients not required

w
→ learnable parameter
→ [1, 1]
→ float32
→ CPU
→ gradients required
```

---

# Practical Training Debugging Checklist

1. Confirm prediction and target shapes.
2. Check for unintended broadcasting.
3. Check dtype and device compatibility.
4. Check `requires_grad` on trainable parameters.
5. Confirm parameters belong to the optimizer.
6. Inspect gradients after `backward()`.
7. Verify gradients are cleared at the correct boundary.
8. Confirm `optimizer.step()` changes parameters.
9. Distinguish pre-update and post-update loss.
10. Track both training and validation curves.
11. Use log scales when large ranges hide behavior.
12. Ensure validation data never supplies training gradients.
13. Restore and verify the best checkpoint.
14. Restart the notebook kernel and run top-to-bottom.
15. Record seeds and important hyperparameters.
16. Remember that a seed improves reproducibility but does not guarantee identical results across every hardware/library combination.

---

# Key Vocabulary

| Term | Meaning |
|---|---|
| Tensor | N-dimensional numerical array |
| Parameter | Value learned during training |
| Hyperparameter | Training choice configured by the practitioner |
| Forward pass | Compute predictions from inputs |
| Loss | Scalar objective measuring error |
| Gradient | Local sensitivity of loss to a parameter |
| Backpropagation | Chain-rule-based gradient computation through a graph |
| Autograd | PyTorch automatic differentiation |
| Optimizer | Algorithm that converts gradients into parameter updates |
| Learning rate | Controls update step size |
| Epoch | One pass through the training dataset |
| Batch | Subset used for one training step |
| Momentum | Decaying optimizer memory of previous directions |
| Validation set | Held-out data used to estimate generalization |
| Overfitting | Training improves while unseen performance deteriorates |
| Underfitting | Model cannot adequately fit the training structure |
| Regularization | Techniques used to constrain training or improve generalization |
| Early stopping | Stop when validation performance no longer improves |
| Weight decay | Parameter shrinkage used as regularization |
| Model capacity | Flexibility of the model |
| Generalization | Performance on unseen data |
| Checkpoint | Saved model state |

---

# Final Mental Model

```text
data
↓
tensor representation
↓
model parameters
↓
forward pass
↓
prediction
↓
loss
↓
backpropagation
↓
gradients
↓
optimizer
↓
parameter update
↓
repeat
↓
validation
↓
generalization analysis
↓
regularization / model selection
```

The most important lesson from Module 1 is:

> **Training a model is not just minimizing training loss. It is learning parameters that generalize to the behavior and data we actually care about.**