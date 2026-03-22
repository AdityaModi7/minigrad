"""
Neural network building blocks built on top of the MiniGrad engine.
Everything here is a pure Python implementation — no numpy, no frameworks.
"""

import random
from minigrad.engine import Value


class Neuron:
    """A single neuron with configurable activation."""

    def __init__(self, n_inputs, activation='relu'):
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(n_inputs)]
        self.b = Value(0, label='b')
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'linear':
            return act
        raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, n_inputs, n_outputs, activation='relu'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-layer perceptron. Last layer uses linear activation by default."""

    def __init__(self, n_inputs, layer_sizes, activations=None):
        sizes = [n_inputs] + layer_sizes
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 1) + ['linear']
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activations[i])
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
