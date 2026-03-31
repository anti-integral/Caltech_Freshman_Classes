from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23).f_relu()",
            "[Value(1.230, grad=1.000), Value(1.230, grad=1.000)]",
            id="simple",
        ),
        pytest.param(
            "Value(-2412.4).f_relu()",
            "[Value(-2412.400, grad=0.000), Value(0.000, grad=1.000)]",
            id="neg",
        ),
        pytest.param(
            "(3 + (Value(23.1))).f_relu()",
            "[Value(3.000, grad=1.000), Value(23.100, grad=1.000), Value(26.100, grad=1.000), Value(26.100, grad=1.000)]",
            id="mixed_outside:1",
        ),
        pytest.param(
            "((Value(32) ** 3) / 3 - Value(5.3)).f_relu()",
            "[Value(-5.300, grad=1.000), Value(-1.000, grad=5.300), Value(0.333, grad=32768.000), Value(3.000, grad=-3640.889), Value(5.300, grad=-1.000), Value(32.000, grad=1024.000), Value(10917.367, grad=1.000), Value(10917.367, grad=1.000), Value(10922.667, grad=1.000), Value(32768.000, grad=0.333)]",
            id="mixed_outside:2",
        ),
        pytest.param(
            "((Value(32) * 3) / 3 + Value(5.3)).f_relu()",
            "[Value(0.333, grad=96.000), Value(3.000, grad=-10.667), Value(3.000, grad=10.667), Value(5.300, grad=1.000), Value(32.000, grad=1.000), Value(32.000, grad=1.000), Value(37.300, grad=1.000), Value(37.300, grad=1.000), Value(96.000, grad=0.333)]",
            id="mixed_outside:3",
        ),
        pytest.param(
            "3 + (Value(32.3) / ((Value(42) * 33).f_relu()))",
            "[Value(0.001, grad=32.300), Value(0.023, grad=1.000), Value(3.000, grad=1.000), Value(3.023, grad=1.000), Value(32.300, grad=0.001), Value(33.000, grad=-0.001), Value(42.000, grad=-0.001), Value(1386.000, grad=-0.000), Value(1386.000, grad=-0.000)]",
            id="mixed_inside:1",
        ),
        pytest.param(
            "(Value(32).f_relu() * 3) / 3 + Value(5.3)",
            "[Value(0.333, grad=96.000), Value(3.000, grad=-10.667), Value(3.000, grad=10.667), Value(5.300, grad=1.000), Value(32.000, grad=1.000), Value(32.000, grad=1.000), Value(32.000, grad=1.000), Value(37.300, grad=1.000), Value(96.000, grad=0.333)]",
            id="mixed_inside:2",
        ),
    ],
)
def test_relu(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
