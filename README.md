# MiniGrad

A tiny autograd engine and neural network library built **from scratch** in pure Python. No NumPy, no PyTorch — just math and backpropagation.

MiniGrad implements the same core algorithm that powers deep learning frameworks like PyTorch: **reverse-mode automatic differentiation** over a dynamically built computation graph.

## What it does

```python
from minigrad import Value

# Build expressions — a computation graph is traced automatically
a = Value(2.0)
b = Value(3.0)
c = a * b + a ** 2  # c = 10.0

# Backpropagate — gradients flow through every operation
c.backward()
print(a.grad)  # 7.0 (dc/da = b + 2a = 3 + 4)
print(b.grad)  # 2.0 (dc/db = a)
```

## Neural Networks

Build and train neural networks using the included `MLP` class:

```python
from minigrad import MLP

# 2 inputs → 16 hidden → 16 hidden → 1 output
model = MLP(2, [16, 16, 1])

# Forward pass
prediction = model([1.0, -2.0])

# Backprop
prediction.backward()

# SGD update
for p in model.parameters():
    p.data -= 0.01 * p.grad
```

## Run the demo

Train a network to classify a moons dataset with an ASCII decision boundary:

```bash
python demo.py
```

```
MiniGrad Neural Network
Architecture: 2 → 16 → 16 → 1
Parameters: 337
Training on 100 samples (moons dataset)
─────────────────────────────────────────────
  epoch   1 | loss 1.1047 | accuracy 44%
  epoch  10 | loss 0.4513 | accuracy 86%
  epoch  20 | loss 0.1889 | accuracy 95%
  epoch  50 | loss 0.0512 | accuracy 100%

  Decision Boundary (ASCII)
  ──────────────────────────────────────────
  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░
  █████████████████░░░░░░░░░░░░░░░░░░░░░░░
  ██████████████████░░░░░░░░░░░░░░░░░░░░░░
  ...
```

## Computation graph visualization

If you have Graphviz installed, you can visualize the computation graph:

```python
from minigrad import Value
from minigrad.viz import draw_graph

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b; c.label = 'c'
c.backward()

dot = draw_graph(c)
dot.render('graph', view=True)
```

## Supported operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| `+` | `a + b` | Both grads += upstream |
| `*` | `a * b` | Cross-multiply rule |
| `**` | `a ** n` | Power rule |
| `/` | `a / b` | Quotient rule |
| `relu()` | `max(0, x)` | Pass-through if positive |
| `tanh()` | `tanh(x)` | `1 - tanh²(x)` |
| `exp()` | `eˣ` | `eˣ` |
| `log()` | `ln(x)` | `1/x` |

## Run tests

```bash
pytest tests/ -v
```

## How it works

1. **Forward pass**: Each operation (`+`, `*`, `relu`, etc.) creates a new `Value` node that remembers its inputs and the operation used. This builds a directed acyclic graph (DAG).

2. **Backward pass**: Starting from the output, `.backward()` topologically sorts the graph, then walks it in reverse, applying the chain rule at each node to compute `∂output/∂node`.

3. **Neural networks**: The `Neuron`, `Layer`, and `MLP` classes compose `Value` objects. Training is just: forward → loss → backward → update weights.

This is literally how PyTorch works under the hood — just at a much smaller scale.

