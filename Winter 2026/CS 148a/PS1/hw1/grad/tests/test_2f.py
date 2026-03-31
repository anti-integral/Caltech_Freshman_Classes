from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23).f_sigmoid()",
            "[Value(0.774, grad=1.000), Value(1.230, grad=0.175)]",
            id="simple",
        ),
        pytest.param(
            "Value(-0.4).f_sigmoid()",
            "[Value(-0.400, grad=0.240), Value(0.401, grad=1.000)]",
            id="neg",
        ),
        pytest.param(
            "(0.3 + (Value(1.1))).f_sigmoid()",
            "[Value(0.300, grad=0.159), Value(0.802, grad=1.000), Value(1.100, grad=0.159), Value(1.400, grad=0.159)]",
            id="mixed_outside:1",
        ),
        pytest.param(
            "((Value(3) ** 2) / 8 - Value(0.8)).f_sigmoid()",
            "[Value(-1.000, grad=0.195), Value(-0.800, grad=0.244), Value(0.125, grad=2.192), Value(0.325, grad=0.244), Value(0.581, grad=1.000), Value(0.800, grad=-0.244), Value(1.125, grad=0.244), Value(3.000, grad=0.183), Value(8.000, grad=-0.034), Value(9.000, grad=0.030)]",
            id="mixed_outside:2",
        ),
        pytest.param(
            "((Value(2) * 3) / 8 + Value(0.1)).f_sigmoid()",
            "[Value(0.100, grad=0.210), Value(0.125, grad=1.259), Value(0.701, grad=1.000), Value(0.750, grad=0.210), Value(0.850, grad=0.210), Value(2.000, grad=0.079), Value(3.000, grad=0.052), Value(6.000, grad=0.026), Value(8.000, grad=-0.020)]",
            id="mixed_outside:3",
        ),
        pytest.param(
            "3 + (Value(32.3) / ((Value(0.1) * 3).f_sigmoid()))",
            "[Value(0.100, grad=-71.785), Value(0.300, grad=-23.928), Value(0.574, grad=-97.883), Value(1.741, grad=32.300), Value(3.000, grad=-2.393), Value(3.000, grad=1.000), Value(32.300, grad=1.741), Value(56.228, grad=1.000), Value(59.228, grad=1.000)]",
            id="mixed_inside:1",
        ),
        pytest.param(
            "(Value(-0.5).f_sigmoid() * 3) / 3 + Value(5.3)",
            "[Value(-0.500, grad=0.235), Value(0.333, grad=1.133), Value(0.378, grad=1.000), Value(0.378, grad=1.000), Value(1.133, grad=0.333), Value(3.000, grad=-0.126), Value(3.000, grad=0.126), Value(5.300, grad=1.000), Value(5.678, grad=1.000)]",
            id="mixed_inside:2",
        ),
    ],
)
def test_sigmoid(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
