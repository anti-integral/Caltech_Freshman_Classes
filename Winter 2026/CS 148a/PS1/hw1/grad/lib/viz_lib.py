"""
Code from: https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb
"""

from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="png", rankdir="LR", show_grad=True):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir, "bgcolor": "transparent"}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label=(
                f"{{ {n.name} | {n.data:.3f} | grad {n.grad:.3f} }}"
                if show_grad
                else f"{{ {n.name} | {n.data:.3f} }}"
            ),
            shape="record",
        )
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    dot.attr(dpi="150")
    return dot
