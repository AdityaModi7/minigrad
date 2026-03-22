"""
Visualization utilities for MiniGrad computation graphs.
Uses Graphviz to render the full DAG of operations and gradients.
"""

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None


def draw_graph(root, format='svg', rankdir='LR'):
    """Render the computation graph rooted at `root` as a Graphviz diagram.

    Args:
        root: A Value node (typically the loss) to trace backward from.
        format: Output format ('svg', 'png', 'pdf').
        rankdir: Layout direction ('LR' for left-to-right, 'TB' for top-to-bottom).

    Returns:
        A Graphviz Digraph object that can be rendered or displayed in a notebook.
    """
    if Digraph is None:
        raise ImportError("Install graphviz: pip install graphviz")

    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    nodes, edges = _trace(root)
    for n in nodes:
        label = f"{{ {n.label + ' | ' if n.label else ''}data {n.data:.4f} | grad {n.grad:.4f} }}"
        dot.node(str(id(n)), label, shape='record')
        if n._op:
            dot.node(str(id(n)) + n._op, n._op, shape='circle',
                     width='0.3', height='0.3', fixedsize='true')
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def _trace(root):
    """Walk the graph and collect all nodes and edges."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def training_plot(losses, title="Training Loss"):
    """Simple ASCII training loss plot for terminal output."""
    if not losses:
        return ""

    max_loss = max(losses)
    min_loss = min(losses)
    height = 15
    width = min(60, len(losses))

    # Resample losses if there are more than width
    if len(losses) > width:
        step = len(losses) / width
        sampled = [losses[int(i * step)] for i in range(width)]
    else:
        sampled = losses
        width = len(sampled)

    loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

    lines = [f"\n  {title}"]
    lines.append(f"  {'─' * (width + 8)}")

    for row in range(height, -1, -1):
        threshold = min_loss + (row / height) * loss_range
        line = f"  {threshold:6.3f} │"
        for val in sampled:
            normalized = (val - min_loss) / loss_range * height
            if normalized >= row:
                line += "█"
            else:
                line += " "
        lines.append(line)

    lines.append(f"  {'':>6} └{'─' * width}")
    lines.append(f"  {'':>6}  0{' ' * (width - 5)}step {len(losses)}")
    lines.append(f"  Final loss: {losses[-1]:.6f}\n")

    return "\n".join(lines)
