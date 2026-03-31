from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23).f_tanh()",
            "[Value(0.843, grad=1.000), Value(1.230, grad=0.290)]",
            id="simple",
        ),
        pytest.param(
            "Value(-0.4).f_tanh()",
            "[Value(-0.400, grad=0.856), Value(-0.380, grad=1.000)]",
            id="neg",
        ),
        pytest.param(
            "(0.3 + (Value(1.1))).f_tanh()",
            "[Value(0.300, grad=0.216), Value(0.885, grad=1.000), Value(1.100, grad=0.216), Value(1.400, grad=0.216)]",
            id="mixed_outside:1",
        ),
        pytest.param(
            "((Value(3) ** 2) / 8 - Value(0.8)).f_tanh()",
            "[Value(-1.000, grad=0.721), Value(-0.800, grad=0.901), Value(0.125, grad=8.113), Value(0.314, grad=1.000), Value(0.325, grad=0.901), Value(0.800, grad=-0.901), Value(1.125, grad=0.901), Value(3.000, grad=0.676), Value(8.000, grad=-0.127), Value(9.000, grad=0.113)]",
            id="mixed_outside:2",
        ),
        pytest.param(
            "((Value(2) * 3) / 8 + Value(0.1)).f_tanh()",
            "[Value(0.100, grad=0.522), Value(0.125, grad=3.135), Value(0.691, grad=1.000), Value(0.750, grad=0.522), Value(0.850, grad=0.522), Value(2.000, grad=0.196), Value(3.000, grad=0.131), Value(6.000, grad=0.065), Value(8.000, grad=-0.049)]",
            id="mixed_outside:3",
        ),
        pytest.param(
            "3 + (Value(32.3) / ((Value(0.1) * 3).f_tanh()))",
            "[Value(0.100, grad=-1044.940), Value(0.291, grad=-380.613), Value(0.300, grad=-348.313), Value(3.000, grad=-34.831), Value(3.000, grad=1.000), Value(3.433, grad=32.300), Value(32.300, grad=3.433), Value(110.877, grad=1.000), Value(113.877, grad=1.000)]",
            id="mixed_inside:1",
        ),
        pytest.param(
            "(Value(-0.5).f_tanh() * 3) / 3 + Value(5.3)",
            "[Value(-1.386, grad=0.333), Value(-0.500, grad=0.786), Value(-0.462, grad=1.000), Value(-0.462, grad=1.000), Value(0.333, grad=-1.386), Value(3.000, grad=-0.154), Value(3.000, grad=0.154), Value(4.838, grad=1.000), Value(5.300, grad=1.000)]",
            id="mixed_inside:2",
        ),
    ],
)
def test_tanh(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
