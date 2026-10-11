# Module 01 — Training Fundamentals

This file is the durable reference for Module 1.

The notebooks contain the executable experiments, saved measurements, plots, and detailed result interpretations:

- [Manual linear regression](01_manual_linear_regression.ipynb) — tensors, gradients, updates, optimizer behavior, and controlled failure cases.
- [Validation and overfitting](02_train_validation_overfitting.ipynb) — generalization, checkpoints, regularization, early stopping, and model capacity.

---

## What This Module Establishes

The core training loop of modern machine learning is:

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

Even when the model later becomes a Transformer with billions of parameters, this basic structure remains recognizable.

Module 1 builds the intuition needed to understand:

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

The main principle is:

> **Understand the mechanics before relying on abstractions.**

---

# Tensor Fundamentals

A **tensor** is an n-dimensional numerical array.

Examples:

```text
scalar       → 0-dimensional tensor
vector       → 1-dimensional tensor
matrix       → 2-dimensional tensor
higher array → n-dimensional tensor
```

A useful mental model is:

```text
Tensor
├── values
├── shape
├── dtype
├── device
└── autograd metadata
```

PyTorch tensors are useful because they support:

- vectorized numerical operations
- matrix multiplication
- accelerator execution
- automatic differentiation
- broadcasting
- multiple numeric precisions

---

## Common Tensor Creation Functions

### `torch.randn`

Creates random values sampled from a standard normal distribution.

```python
w = torch.randn(1, 1)
```

Conceptually:

```text
rand → random
n    → normal distribution
```

Possible result:

```text
tensor([[-0.5672]])
```

Values can be positive or negative and are centered around zero.

---

### `torch.zeros`

Creates a tensor filled with zeros.

```python
b = torch.zeros(1)
```

Result:

```text
tensor([0.])
```

Initialization and gradient tracking are separate concepts.

For example:

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

For example:

```text
[100, 1]
```

might mean:

```text
100 examples
1 feature per example
```

Later:

```text
[8, 512, 768]
```

might mean:

```text
8 sequences
512 token positions
768 hidden features per token
```

The numbers alone do not define the semantics.

---

## Shape Reasoning

For manual linear regression:

| Tensor | Role | Shape | Gradient role |
|---|---|---|---|
| `x` | Input data | `[N, 1]` | Fixed data |
| `y` | Targets | `[N, 1]` | Fixed data |
| `w` | Weight | `[1, 1]` | Learnable parameter |
| `b` | Bias | `[1]` | Learnable parameter |
| `y_pred` | Predictions | `[N, 1]` | Derived through graph |
| `loss` | Objective | `[]` | Scalar derived through graph |

The forward computation:

```python
y_pred = x @ w + b
```

uses:

```text
[N, 1] @ [1, 1]
→ [N, 1]
```

and then broadcasts:

```text
b: [1]
```

across all `N` rows.

---

## Broadcasting

Broadcasting allows PyTorch to apply compatible smaller tensors across larger tensors.

Example:

```text
prediction: [100, 1]
bias:       [1]
```

The same bias value is added to each row.

Broadcasting is convenient, but it can also hide bugs.

For example:

```text
[N, 1] - [N]
```

can produce:

```text
[N, N]
```

rather than one error per example.

Always compare prediction and target shapes explicitly.

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

These are different operations.

For matrix multiplication, inner dimensions must match:

```text
[32, 128] @ [128, 256]
→ [32, 256]
```

but:

```text
[32, 128] @ [64, 256]
```

is invalid because:

```text
128 != 64
```

---

## dtype

Common PyTorch dtypes include:

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
training speed
numerical stability
hardware compatibility
```

For intuition:

```text
1 billion parameters
× 4 bytes per FP32 parameter
≈ 4 GB
```

and that is before:

```text
gradients
optimizer state
activations
```

---

## Device

Possible devices include:

```text
cpu
mps
cuda
```

Move a tensor using:

```python
x = x.to(device)
```

Important:

```python
x.to(device)
```

returns a tensor.

Use the returned value.

Tensors participating in one operation generally need compatible dtypes and to live on the same device.

The Module 1 experiments intentionally use CPU tensors even though the environment supports MPS.

---

# Parameters and Hyperparameters

A **parameter** is learned from data.

Examples:

```text
w
b
neural-network weights
neural-network biases
```

A **hyperparameter** is chosen by us.

Examples:

```text
learning rate
number of epochs
momentum
weight decay
model size
batch size
early-stopping patience
```

Useful distinction:

```text
parameter
→ learned during training

hyperparameter
→ configures how training happens
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

Why?

```text
x, y
→ observed data
→ not being optimized

w, b
→ learnable parameters
→ should receive gradients
```

Important distinction:

```text
requires_grad=True
→ track operations

loss.backward()
→ actually compute gradients

optimizer.step()
→ actually modify parameters
```

`requires_grad=True` does not update anything by itself.

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
        │
        ├──> y_pred ───> loss
        │
b ──────┘
```

Then:

```python
loss.backward()
```

traverses that graph backward using the chain rule.

PyTorch computes:

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

---

## Leaf vs Derived Tensors

Parameters such as:

```python
w
b
```

are typically **leaf tensors**.

Derived tensors such as:

```text
y_pred
loss
```

participate in the graph but are not parameters.

A tensor can have:

```text
requires_grad = True
```

without itself being something the optimizer updates.

---

# Linear Regression Model

The synthetic dataset follows approximately:

$$
y = 3x + 2 + \epsilon
$$

where:

```text
3 → true slope
2 → true intercept
ε → noise
```

The model is:

$$
\hat{y} = wx + b
$$

where:

```text
w → weight / slope
b → bias / intercept
```

The weight controls the line's tilt.

The bias controls its vertical position.

When:

$$
x = 0
$$

then:

$$
\hat{y} = b
$$

which is why `b` corresponds to the intercept.

---

# Loss

The Module 1 regression experiments use Mean Squared Error:

$$
L
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
$$

Why square the errors?

```text
positive and negative errors cannot cancel
+
large errors receive stronger penalties
```

The loss is a scalar:

```text
shape = []
```

A loss tells us how wrong the model is under a particular objective.

It does not directly tell us the correct parameter values.

---

# Manual Gradient Derivation

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

the gradient with respect to the weight is:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{N}
\sum_i
(\hat{y}_i-y_i)x_i
$$

and with respect to the bias:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{N}
\sum_i
(\hat{y}_i-y_i)
$$

The manual gradients matched PyTorch autograd to floating-point precision.

This demonstrates that:

```python
loss.backward()
```

is automating the same calculus.

---

# What a Gradient Means

A gradient is a **local sensitivity**.

It answers:

> If this parameter changes slightly, how does the loss change?

Suppose:

```text
w.grad < 0
```

Then increasing `w` locally reduces the loss.

Suppose:

```text
w.grad > 0
```

Then decreasing `w` locally reduces the loss.

The gradient does not say:

```text
set w = 3
```

It only provides local information about the loss surface.

---

# Gradient Descent

Gradient descent updates a parameter using:

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

A negative gradient causes the parameter to increase because:

```text
parameter
-
negative number
=
parameter increase
```

A positive gradient causes the parameter to decrease.

---

# Learning Rate

The learning rate controls update size.

Conceptually:

```text
gradient
→ direction + local steepness

learning rate
→ how far we move
```

Observed behavior:

```text
very small learning rate
→ correct direction
→ slow convergence

reasonable learning rate
→ stable convergence

very large learning rate
→ overshooting
→ divergence
```

A large learning rate can cause:

```text
overshoot
↓
larger gradient
↓
larger update
↓
more overshoot
↓
numerical explosion
```

Optimizer behavior always depends on its hyperparameters.

---

# Training Loop Mental Model

The canonical loop is:

```text
forward
↓
loss
↓
backward
↓
gradient
↓
update
↓
clear gradients
↓
repeat
```

Equivalent PyTorch code:

```python
for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)

    loss.backward()
    optimizer.step()
```

Line by line:

```text
optimizer.zero_grad()
→ clear previous gradients

model(x_train)
→ forward pass

loss_fn(...)
→ compute scalar objective

loss.backward()
→ compute gradients

optimizer.step()
→ update parameters
```

---

# Epoch, Batch, and Full-Batch Training

An **epoch** is one complete pass through the training dataset.

In the early regression experiments:

```text
all training examples
→ one forward pass
→ one backward pass
→ one update
```

so each epoch contains one full-batch update.

Later, with mini-batches:

```text
dataset
↓
batch 1 → update
batch 2 → update
batch 3 → update
...
↓
all examples seen once
↓
1 epoch
```

The Module 1 experiments use full batches.

---

# Gradient Clearing and Accumulation

PyTorch accumulates gradients by default.

Suppose:

```text
backward #1 → g1
backward #2 → g2
backward #3 → g3
```

Without clearing:

```text
after #1 → g1
after #2 → g1 + g2
after #3 → g1 + g2 + g3
```

For ordinary training, we therefore use:

```python
optimizer.zero_grad()
```

or manually:

```python
w.grad.zero_()
b.grad.zero_()
```

---

## Why accidental accumulation is a problem

A gradient describes the loss surface at a particular parameter state.

Old gradients were computed using older model parameters.

If they remain in `.grad`, future updates no longer correspond to ordinary gradient descent.

Observed behavior:

```text
clear gradients
→ smooth convergence

do not clear gradients
→ oscillation / instability
```

---

## Intentional Gradient Accumulation

Gradient accumulation can also be deliberate.

For example:

```text
microbatch 1
microbatch 2
microbatch 3
microbatch 4
↓
one optimizer update
```

This is useful when hardware cannot fit the desired effective batch size.

Important distinction:

```text
unintentional accumulation
→ bug

controlled accumulation
→ training technique
```

---

# Manual Updates and `torch.no_grad()`

A manual parameter update:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

uses `torch.no_grad()` because the parameter update itself should not become part of the differentiable computation graph.

We want to differentiate:

```text
prediction
→ loss
```

not:

```text
previous parameter update
→ current parameter update
→ future parameter update
```

After parameters change, recompute predictions if you want to measure the new loss.

---

# `nn.Module`

Instead of manually managing:

```python
w
b
```

PyTorch models usually subclass or compose:

```python
nn.Module
```

An `nn.Module` organizes:

```text
parameters
layers
forward computation
nested components
```

For linear regression:

```python
model = nn.Linear(1, 1)
```

replaces the manually managed affine transformation.

---

# `nn.Linear`

`nn.Linear(in_features, out_features)` computes:

$$
y = xW^T+b
$$

The stored weight shape is:

```text
[out_features, in_features]
```

For:

```python
nn.Linear(1, 1)
```

the parameter shapes are:

```text
weight → [1, 1]
bias   → [1]
```

This storage convention differs from some manually defined matrices shaped:

```text
[in_features, out_features]
```

so always inspect shapes.

---

# `nn.MSELoss`

PyTorch provides:

```python
loss_fn = nn.MSELoss()
```

which by default computes the mean squared error.

For the Module 1 experiment:

```python
loss_fn(y_pred, y)
```

matched:

```python
((y_pred - y) ** 2).mean()
```

---

# Optimizer Abstraction

Before optimizers:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

With PyTorch:

```python
optimizer.step()
```

The optimizer:

```text
does not compute gradients
```

It reads gradients already produced by:

```python
loss.backward()
```

and applies its update rule.

---

# Optimizer Concepts

| Optimizer | Update information | Practical implication |
|---|---|---|
| SGD | Current gradient | Simple, direct baseline |
| SGD + momentum | Current gradient + decaying velocity | Can accelerate and overshoot |
| Adam | Running first and second moments | Adaptive parameter-wise updates |
| AdamW | Adam-style adaptation + decoupled weight decay | Common in Transformer training |

---

# SGD

Basic SGD performs:

$$
\theta_{t+1}
=
\theta_t
-
\eta g_t
$$

where:

```text
g_t → current gradient
```

On the simple convex regression problem, SGD performed extremely well.

Important lesson:

> A more sophisticated optimizer is not automatically better on every problem.

---

# Momentum

Momentum adds optimizer state.

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

where:

```text
v_t → velocity
β   → momentum coefficient
```

Conceptually:

```text
current gradient
+
previous velocity
↓
new velocity
↓
parameter update
```

Momentum can:

- accelerate movement in consistent directions
- reduce noisy zig-zagging
- overshoot minima
- interact strongly with learning rate

Observed behavior in Module 1:

```text
SGD without momentum
→ smooth convergence

momentum = 0.9 with same learning rate
→ faster early movement
→ overshoot
→ oscillation
```

Momentum is not guaranteed to improve every setup.

---

# Momentum vs Gradient Accumulation

Accidental accumulation:

```text
g1
g1 + g2
g1 + g2 + g3
```

Momentum:

```text
recent gradients
→ strong influence

older gradients
→ progressively decaying influence
```

Momentum is controlled optimizer state.

Gradient accumulation is direct addition into `.grad`.

They are different mechanisms.

---

# Adam

Adam tracks running estimates of:

```text
first moment
→ roughly gradient direction history

second moment
→ roughly squared gradient magnitude history
```

Conceptually:

```text
gradient
↓
track direction history
+
track magnitude history
↓
adaptive parameter update
```

Adam is useful when:

- parameter gradients have different scales
- optimization is noisy
- models are large
- gradients are irregular

Its learning-rate scale is not directly comparable to SGD.

---

# AdamW

AdamW combines Adam-style adaptive optimization with **decoupled weight decay**.

Conceptually:

```text
adaptive gradient update
+
parameter shrinkage
```

AdamW is commonly used for Transformer and LLM training.

Important nuance:

AdamW's weight decay is not identical to simply adding an L2 penalty to the loss and passing that derivative through Adam.

The Module 1 optimizer comparison also used AdamW's default nonzero weight decay, so the comparison was not purely optimizer-state-only.

---

# Train / Validation Split

Training data is used for:

```text
forward pass
loss
backward pass
parameter update
```

Validation data is used for:

```text
evaluation only
```

Validation data should not supply gradients used to update parameters.

Conceptually:

```text
training set
→ learn parameters

validation set
→ measure generalization
```

---

## Why Shuffle Before Splitting?

The synthetic inputs were sorted.

If we had used:

```text
first 80% → training
last 20% → validation
```

the two sets would cover different regions of the input space.

Random shuffling produced representative subsets from the same distribution.

But random splitting is not universally correct.

Examples:

```text
time series
→ preserve temporal ordering

user-based data
→ split by user when appropriate

documents from same source
→ grouped split may prevent leakage
```

The split should reflect the intended deployment problem.

---

# Generalization

Generalization means:

> The model performs usefully on examples it did not train on.

Training loss answers:

> How well does the model fit its training examples?

Validation loss answers:

> How well does the learned behavior transfer to unseen examples?

The goal of machine learning is generally not:

```text
minimize training loss at any cost
```

It is closer to:

```text
learn useful patterns
↓
perform well on unseen data
```

---

# Generalization Gap

The difference between training and validation performance is often called the **generalization gap**.

A gap alone does not prove overfitting.

Consider:

```text
noise
sample difficulty
dataset size
curve behavior over time
```

The strongest overfitting signal is usually:

```text
training loss continues decreasing

while

validation loss stops improving or increases
```

---

# Overfitting

Overfitting occurs when a model increasingly specializes to the training data while generalization worsens.

Typical pattern:

```text
early training
→ learn useful signal

later training
→ increasingly fit noise / sample-specific details

training loss
→ continues decreasing

validation loss
→ stops improving or rises
```

In the deliberate overfitting experiment:

```text
training loss became nearly zero
while
validation loss remained much larger
```

The best validation model occurred very early compared with the final training epoch.

---

# Underfitting

Underfitting occurs when the model is too limited or insufficiently optimized to capture the underlying structure.

Typical pattern:

```text
training loss high
validation loss high
```

Possible causes include:

```text
too little model capacity
insufficient training
poor optimization
excessive regularization
```

---

# Model Capacity

Model capacity describes how flexible a model is.

Observed Module 1 behavior:

```text
small model
→ too little capacity
→ underfitting

medium model
→ enough capacity
→ best validation performance

large model
→ near-zero training loss
→ stronger overfitting
```

The model with the lowest training loss was not the model with the best validation performance.

Important lesson:

> The best model is not necessarily the largest model.

---

# Bias–Variance Intuition

Very roughly:

```text
low-capacity model
→ higher bias
→ lower flexibility

high-capacity model
→ lower bias
→ potentially higher variance
```

Too little flexibility can miss real structure.

Too much flexibility can make the model sensitive to a small training sample.

The goal is a balance that generalizes well.

---

# Early Stopping

Early stopping monitors validation performance during training.

Conceptually:

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

In the Module 1 experiment, the best validation checkpoint occurred much earlier than the stopping epoch.

---

## Patience

Patience controls how long training may continue without improvement.

Example:

```text
patience = 50
```

means:

> Stop after 50 consecutive epochs without validation improvement.

Too little patience can stop training prematurely.

Too much patience can waste compute.

---

## Saving the Best Checkpoint

Use:

```python
copy.deepcopy(model.state_dict())
```

to make an in-memory copy.

Keeping references to live tensors can allow later updates to overwrite the intended checkpoint.

After stopping:

```python
model.load_state_dict(best_model_state)
```

restores the best model.

---

# `model.train()` vs `model.eval()` vs `torch.no_grad()`

These are different concepts.

### `model.train()`

Enables training behavior for mode-dependent layers.

Examples:

```text
dropout
batch normalization
```

### `model.eval()`

Enables evaluation behavior for mode-dependent layers.

It does **not** disable gradient tracking.

### `torch.no_grad()`

Disables autograd graph construction.

Typical validation pattern:

```python
model.eval()

with torch.no_grad():
    val_pred = model(x_val)
```

---

# Weight Decay

Weight decay discourages large parameter values.

A simplified regularized objective is often described as:

$$
L_{\text{total}}
=
L_{\text{data}}
+
\lambda \lVert W \rVert^2
$$

where:

```text
λ → regularization strength
```

Conceptually:

```text
fit the data
+
discourage unnecessarily large parameters
```

Observed Module 1 behavior:

```text
more weight decay
→ higher training loss
→ less aggressive memorization
→ only small validation improvement
```

Weight decay did not eliminate overfitting.

Regularization is not guaranteed to improve every experiment.

---

# Early Stopping vs Weight Decay

These address overfitting differently.

Early stopping:

```text
controls how long training continues
```

Weight decay:

```text
changes how freely parameters can grow
```

So:

```text
early stopping
→ temporal regularization

weight decay
→ parameter regularization
```

They can be used together.

---

# Core Experiment Takeaways

## Learning Rate

```text
too small
→ slow convergence

reasonable
→ stable convergence

too large
→ divergence
```

---

## Gradient Clearing

```text
clear every step
→ intended gradient descent

forget to clear
→ gradients accumulate
→ optimization changes
```

---

## Manual vs Autograd

```text
manual derivative
≈
PyTorch autograd derivative
```

Autograd automates calculus; it does not replace the underlying mathematics.

---

## `nn.Module` and Optimizers

The abstractions did not change the core training mechanics.

```text
manual w, b
→ model.parameters()

x @ w + b
→ model(x)

manual MSE
→ nn.MSELoss()

manual parameter update
→ optimizer.step()

manual gradient clearing
→ optimizer.zero_grad()
```

---

## Optimizers

```text
SGD
→ simple and effective on the toy problem

Adam
→ adaptive gradient-history-based updates

AdamW
→ Adam-style updates + decoupled weight decay

Momentum
→ adds controlled velocity
→ can accelerate
→ can overshoot
```

There is no universal optimizer ranking.

---

## Generalization

```text
low training loss
≠
automatically good model
```

Validation performance matters.

---

## Overfitting

```text
training loss ↓
validation loss ↑
```

is a classic warning pattern.

---

## Early Stopping

```text
best checkpoint
may occur long before
final epoch
```

---

## Weight Decay

```text
reduced training fit
but
did not automatically solve overfitting
```

---

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

When inspecting a tensor:

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

1. Confirm prediction and target shapes exactly match the intended semantics.
2. Watch for unintended broadcasting.
3. Check dtype and device compatibility.
4. Check `requires_grad` on learnable parameters.
5. Confirm model parameters are registered with the optimizer.
6. Inspect gradients after `backward()`.
7. Verify gradients are cleared at the intended boundary.
8. Confirm `optimizer.step()` actually changes parameters.
9. Distinguish pre-update loss from post-update loss.
10. Track both training and validation loss.
11. Use a log scale when large differences hide useful behavior.
12. Confirm validation data never contributes parameter-update gradients.
13. Restore the recorded best checkpoint and verify its validation loss.
14. Restart the kernel and run the notebook top-to-bottom.
15. Record random seeds and hyperparameters.
16. Remember that a fixed seed improves reproducibility but does not guarantee bit-identical results across all hardware, library versions, or platforms.

---

# Key Vocabulary

| Term | Meaning |
|---|---|
| Tensor | N-dimensional numerical array |
| Parameter | Value learned during training |
| Hyperparameter | Training choice configured by the practitioner |
| Forward pass | Compute predictions from inputs |
| Loss | Scalar objective measuring model error |
| Gradient | Local sensitivity of loss to a parameter |
| Backpropagation | Efficient application of the chain rule through the computational graph |
| Autograd | PyTorch's automatic differentiation system |
| Optimizer | Algorithm that converts gradients into parameter updates |
| Learning rate | Controls update step size |
| Epoch | One pass through the training dataset |
| Batch | Subset of examples used for one training step |
| Momentum | Decaying optimizer memory of previous update direction |
| Validation set | Held-out data used to evaluate generalization |
| Overfitting | Training improves while unseen performance deteriorates |
| Underfitting | Model cannot adequately fit even the training structure |
| Regularization | Techniques that constrain or alter training to improve generalization |
| Early stopping | Stop when validation performance no longer improves |
| Weight decay | Parameter shrinkage used as regularization |
| Model capacity | Flexibility of the model to represent functions |
| Generalization | Performance on unseen data |
| Checkpoint | Saved model state at a particular training point |

---

# Final Mental Model

The entire module can be compressed into:

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

The most important lesson is:

> **Training a model is not just minimizing training loss. It is learning parameters that generalize to the behavior and data we actually care about.**