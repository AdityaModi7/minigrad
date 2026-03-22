"""
MiniGrad Demo — trains a small neural network to learn a circular decision
boundary, with a live ASCII training plot in the terminal.

Run: python demo.py
"""

import random
from minigrad import Value, MLP
from minigrad.viz import training_plot

# ──────────────────────────────────────────────
# Generate a "moons" dataset (two interleaving half-circles)
# ──────────────────────────────────────────────
random.seed(42)

def make_moons(n=100, noise=0.1):
    """Generate two interleaving half-circles of 2D points."""
    import math
    data, labels = [], []
    for i in range(n):
        if i < n // 2:
            angle = math.pi * i / (n // 2)
            x = math.cos(angle) + random.gauss(0, noise)
            y = math.sin(angle) + random.gauss(0, noise)
            labels.append(1.0)
        else:
            angle = math.pi * (i - n // 2) / (n // 2)
            x = 1 - math.cos(angle) + random.gauss(0, noise)
            y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
            labels.append(-1.0)
        data.append([x, y])
    return data, labels

X, y = make_moons(100, noise=0.15)

# ──────────────────────────────────────────────
# Build a small MLP: 2 inputs → 16 → 16 → 1 output
# ──────────────────────────────────────────────
model = MLP(2, [16, 16, 1])
print(f"MiniGrad Neural Network")
print(f"Architecture: 2 → 16 → 16 → 1")
print(f"Parameters: {len(model.parameters())}")
print(f"Training on {len(X)} samples (moons dataset)")
print(f"{'─' * 45}")

# ──────────────────────────────────────────────
# Training loop with hinge loss (SVM-style)
# ──────────────────────────────────────────────
losses = []
lr = 0.01

for epoch in range(50):
    # Forward pass — compute predictions and loss
    preds = [model(x) for x in X]
    data_loss = sum(
        (1 + -yi * pi).relu()
        for yi, pi in zip(y, preds)
    ) * (1.0 / len(y))

    # L2 regularization
    reg_loss = 0.0001 * sum(p * p for p in model.parameters())
    total_loss = data_loss + reg_loss

    # Backward pass
    model.zero_grad()
    total_loss.backward()

    # Update weights (SGD)
    for p in model.parameters():
        p.data -= lr * p.grad

    # Track progress
    loss_val = total_loss.data
    losses.append(loss_val)

    accuracy = sum(
        (1 if pi.data > 0 else -1) == int(yi)
        for yi, pi in zip(y, preds)
    ) / len(y)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:3d} | loss {loss_val:.4f} | accuracy {accuracy:.0%}")

# ──────────────────────────────────────────────
# Show training curve
# ──────────────────────────────────────────────
print(training_plot(losses))

# ──────────────────────────────────────────────
# Show the decision boundary as ASCII art
# ──────────────────────────────────────────────
print("  Decision Boundary (ASCII)")
print(f"  {'─' * 42}")

for row in range(20, -1, -1):
    line = "  "
    yy = -0.5 + row * 2.0 / 20
    for col in range(40):
        xx = -0.5 + col * 2.5 / 40
        pred = model([xx, yy])
        val = pred.data
        if val > 0.5:
            line += "█"
        elif val > 0:
            line += "▓"
        elif val > -0.5:
            line += "░"
        else:
            line += " "
    print(line)

# Overlay data points
print(f"\n  Legend: █▓ = class +1 | ░  = class -1")
print(f"  The network learned to separate the two moons!\n")
