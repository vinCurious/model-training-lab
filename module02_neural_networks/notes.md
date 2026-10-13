# Module 2 — Neural Networks, Initialization, and Deep Training

This module builds from linear models into deeper neural networks and focuses on three questions:

1. Why do neural networks need nonlinear activations?
2. Why can deep networks be difficult to train?
3. How do initialization, normalization, and residual connections improve trainability?

The notebooks contain the full experiments and measured results. These notes are a compact reference for the main concepts.

---

## 1. From Linear Models to Neural Networks

A linear model has the form:

$$
\hat{y}=xW+b
$$

Stacking linear transformations without nonlinear activations does not increase functional expressiveness.

For example:

$$
h=xW_1+b_1
$$

$$
\hat{y}=hW_2+b_2
$$

can be rewritten as:

$$
\hat{y}=xW'+b'
$$

for some equivalent $W'$ and $b'$.

Therefore:

```text
Linear
→ Linear
→ Linear
```

is still equivalent to:

```text
one Linear transformation
```

### Why nonlinear activations matter

A nonlinear activation prevents those transformations from collapsing into one linear function.

For ReLU:

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

A basic MLP becomes:

```text
input
↓
Linear
↓
ReLU
↓
Linear
↓
output
```

This allows the network to represent nonlinear functions.

---

## 2. Hidden Representations

A hidden layer transforms the original features into learned intermediate features.

For example:

```text
input
[batch, 1]

↓

Linear(1, 16)

↓

hidden representation
[batch, 16]
```

The hidden features are not manually designed.

They are learned through optimization.

A shallow ReLU network can be written as:

$$
\hat{y}
=
\sum_j
W_{2,j}
\operatorname{ReLU}(xW_{1,j}+b_{1,j})
+b_2
$$

Each hidden ReLU unit behaves like a learned piecewise-linear feature.

Multiple simple features can combine to approximate a complex nonlinear function.

---

## 3. Tensor Shapes in `nn.Linear`

PyTorch stores:

```text
nn.Linear(in_features, out_features)
```

weights with shape:

```text
[out_features, in_features]
```

and computes:

$$
y=xW^T+b
$$

Example:

```text
input              [200, 1]
Linear(1, 16)
weight             [16, 1]
output             [200, 16]
```

Tensor-shape reasoning is essential when building deeper networks.

---

## 4. Expressivity vs Trainability

Two different questions should be separated:

```text
expressivity
→ can the model represent the function?

trainability
→ can optimization find useful parameters?
```

A network may theoretically represent a solution while still being difficult to train.

---

## 5. Forward Signal Scale

For one neuron:

$$
z=\sum_{i=1}^{n}w_ix_i
$$

If inputs and weights are approximately independent:

$$
\operatorname{Var}(z)
\approx
n\operatorname{Var}(w)\operatorname{Var}(x)
$$

where:

```text
n = fan-in
```

This explains why initialization scale should depend on layer width.

Poor initialization can cause signals to shrink or grow repeatedly through depth.

---

## 6. Vanishing and Exploding Gradients

Backpropagation repeatedly applies the chain rule.

Conceptually:

```text
gradient
× local derivative
× weight
× local derivative
× weight
× ...
```

If many factors are smaller than one:

```text
gradient → very small
```

This is a vanishing-gradient regime.

If many factors are large:

```text
gradient → very large
```

This is an exploding-gradient regime.

Initialization influences both forward activation scale and backward gradient scale.

---

## 7. ReLU and He Initialization

ReLU removes negative activations:

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

For ReLU-like networks, He/Kaiming initialization commonly uses:

$$
\operatorname{Var}(W)
\approx
\frac{2}{\text{fan-in}}
$$

or:

$$
\operatorname{Std}(W)
\approx
\sqrt{\frac{2}{\text{fan-in}}}
$$

The goal is not to force every layer to have standard deviation exactly equal to one.

The goal is to avoid systematic signal collapse or explosion through depth.

Typical pairing:

```text
ReLU / LeakyReLU
→ He / Kaiming initialization
```

---

## 8. `tanh`, Saturation, and Xavier Initialization

`tanh` is bounded:

$$
-1\leq\tanh(z)\leq1
$$

Its derivative is:

$$
\frac{d}{dz}\tanh(z)
=
1-\tanh^2(z)
$$

When $|z|$ is large:

$$
\tanh(z)\approx\pm1
$$

and:

$$
\frac{d}{dz}\tanh(z)\approx0
$$

This is saturation.

A bounded activation can therefore still have poor gradient flow.

### Xavier / Glorot initialization

For activations such as `tanh`, Xavier initialization commonly uses:

$$
\operatorname{Var}(W)
=
\frac{2}
{\text{fan-in}+\text{fan-out}}
$$

The goal is to keep activations in a useful, non-saturated regime and maintain reasonable signal propagation.

Typical pairing:

```text
tanh
→ Xavier / Glorot initialization
```

The deeper principle is:

> Initialization should match the network structure and activation behavior.

---

## 9. Gradient Magnitude Is Not Training Quality

A large gradient is not automatically good.

It may represent:

```text
strong learning signal
```

or:

```text
instability / overshooting
```

Likewise, a small gradient may indicate:

```text
vanishing gradients
```

or simply:

```text
small useful updates in a well-conditioned region
```

Gradient diagnostics should always be interpreted alongside:

- loss behavior
- validation performance
- numerical stability
- activation statistics
- optimization trajectory

---

## 10. Layer Normalization

For a hidden vector:

$$
h=[h_1,\dots,h_d]
$$

LayerNorm computes statistics across its features:

$$
\mu
=
\frac{1}{d}
\sum_i h_i
$$

$$
\sigma^2
=
\frac{1}{d}
\sum_i(h_i-\mu)^2
$$

and normalizes:

$$
\hat{h}_i
=
\frac{h_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

It then applies learned scale and shift parameters:

$$
y_i
=
\gamma_i\hat{h}_i+\beta_i
$$

So LayerNorm does not permanently force hidden representations to mean zero and standard deviation one.

`gamma` and `beta` allow the network to learn a useful scale and offset.

LayerNorm changes the geometry and scale of the optimization problem.

It does not guarantee:

```text
larger gradients
smoother loss
lower final loss
```

---

## 11. Residual Connections

A normal transformation computes:

$$
h_{l+1}=F(h_l)
$$

A residual block computes:

$$
h_{l+1}=h_l+F(h_l)
$$

The identity path provides a direct route for information and gradients.

The derivative includes:

$$
\frac{\partial h_{l+1}}
{\partial h_l}
=
1+
\frac{\partial F(h_l)}
{\partial h_l}
$$

The explicit `1` means gradients do not depend entirely on the learned branch.

However:

> Residual connections do not automatically control residual-branch scale.

Skip connections alone do not guarantee easy optimization.

---

## 12. Pre-Normalized Residual Blocks

A useful design is:

$$
h_{l+1}
=
h_l
+
F(\operatorname{LayerNorm}(h_l))
$$

Conceptually:

```text
                 ┌───────────────┐
                 │               │
h ───────────────┼────────────►  +
                 │               ↑
                 ↓               │
             LayerNorm           │
                 ↓               │
             transformation ─────┘
```

This combines two mechanisms:

```text
LayerNorm
→ controls the input to the learned branch

Residual path
→ preserves a direct identity route
```

These ideas appear repeatedly in modern Transformer architectures.

---

## 13. Parameter Count Sanity Checks

A linear layer contains:

$$
\text{in features}\times\text{out features}
+
\text{out features}
$$

parameters.

The second term is the bias.

`ReLU` contains no learnable parameters.

`LayerNorm(d)` typically contains:

```text
d gamma parameters
+
d beta parameters
=
2d learnable parameters
```

For `LayerNorm(64)`:

```text
64 + 64 = 128 parameters
```

In the PreNorm residual model:

```text
4 block LayerNorms × (64 gamma + 64 beta)
= 4 × 128
= 512

final LayerNorm
= 128

extra normalization parameters
= 640
```

This matches the difference between:

```text
Plain residual model     = 33,473 parameters
PreNorm residual model   = 34,113 parameters
```

because:

```text
33,473 + 640 = 34,113
```

Parameter count is useful for debugging and architecture comparison, but:

> More parameters do not automatically imply better optimization or better generalization.

The earlier residual model also had more parameters than the plain MLP and still performed substantially worse.

The organization of the parameters and the resulting optimization dynamics matter.

---

## 14. What the Architecture Experiments Showed

The deeper-network experiments compared four architectures on the same nonlinear regression task.

Approximate best validation losses were:

```text
Plain MLP                ≈ 0.02482
LayerNorm MLP            ≈ 0.01232
Residual MLP             ≈ 0.08057
PreNorm Residual MLP     ≈ 0.01253
```

The plain MLP trained successfully, but its gradients varied substantially throughout optimization.

The LayerNorm model reached the lowest single validation point, but its training trajectory remained fairly oscillatory.

The residual-only model performed worse despite having skip connections and more parameters.

Its residual branches began in a relatively aggressive numerical regime.

The PreNorm residual model combined:

```text
normalization
+
identity path
```

and achieved strong validation performance with a comparatively steady optimization trajectory.

The key lesson is not that one architecture is universally best.

It is:

> Architectural components interact, and their usefulness depends on how they are combined.

---

## 15. Small Gradients Are Not Automatically Bad

The PreNorm residual model often had smaller middle-layer gradients than the other architectures.

For example, at epoch 400:

```text
Plain MLP              ≈ 0.0263
LayerNorm MLP          ≈ 0.00252
Residual MLP           ≈ 0.0154
PreNorm Residual MLP   ≈ 0.000057
```

Despite this, the PreNorm residual model trained effectively.

This reinforces an important principle:

```text
large gradient
≠ automatically good

small gradient
≠ automatically bad
```

A large gradient may indicate strong signal, but it may also indicate instability or poorly scaled representations.

A small gradient may indicate vanishing gradients, but it may also represent small, useful updates in a well-conditioned region.

What matters is the broader training behavior:

```text
Are gradients nonzero?
Are they numerically stable?
Does training loss decrease?
Does validation performance improve?
Does optimization remain controllable?
```

Gradient magnitude should never be interpreted in isolation.

---

## 16. Practical Diagnostics for Deep Networks

When training becomes unstable, inspect:

```text
1. training loss
2. validation loss
3. activation mean/std across depth
4. gradient magnitudes across depth
5. saturation percentage for bounded activations
6. initialization strategy
7. learning rate
8. normalization placement
9. residual-branch scale
10. parameter count and architecture
```

Avoid diagnosing training from one metric alone.

---

## Key Vocabulary

```text
MLP
hidden layer
hidden representation
activation function
ReLU
tanh
piecewise linear
fan-in
fan-out
vanishing gradient
exploding gradient
saturation
He initialization
Kaiming initialization
Xavier initialization
Glorot initialization
LayerNorm
residual connection
skip connection
identity path
PreNorm
gradient flow
```

---

## Final Mental Model

A useful progression is:

```text
Linear layers
→ transform representations

Nonlinear activations
→ create expressive nonlinear models

Initialization
→ gives signals and gradients a healthy starting scale

Normalization
→ helps control intermediate representations during training

Residual connections
→ provide direct paths through deep networks

PreNorm residual structure
→ combines controlled learned transformations with an identity stream
```

Deep learning is therefore not simply:

```text
add more layers
```

It is about designing a system where:

```text
information can flow forward
+
gradients can flow backward
+
optimization remains numerically useful
```

That foundation will become especially important when we move into language models and Transformers.