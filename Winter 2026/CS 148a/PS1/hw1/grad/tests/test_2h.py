from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23).f_relu()",
            "[Value(1.230, grad=1.000), Value(1.230, grad=1.000)]",
            id="simple:1",
        ),
        pytest.param(
            "Value(1.23) / Value(23.42)",
            "[Value(0.043, grad=1.230), Value(0.053, grad=1.000), Value(1.230, grad=0.043), Value(23.420, grad=-0.002)]",
            id="simple:2",
        ),
        pytest.param(
            "Value(-1.0) / Value(32.23)",
            "[Value(-1.000, grad=0.031), Value(-0.031, grad=1.000), Value(0.031, grad=-1.000), Value(32.230, grad=0.001)]",
            id="neg:1",
        ),
        pytest.param(
            "Value(-43.0) / Value(-342.254)",
            "[Value(-342.254, grad=0.000), Value(-43.000, grad=-0.003), Value(-0.003, grad=-43.000), Value(0.126, grad=1.000)]",
            id="neg:2",
        ),
        pytest.param(
            "Value(-2.4).f_tanh()",
            "[Value(-2.400, grad=0.032), Value(-0.984, grad=1.000)]",
            id="neg:3",
        ),
        pytest.param(
            "3 / Value(3.2)",
            "[Value(0.312, grad=3.000), Value(0.938, grad=1.000), Value(3.000, grad=0.312), Value(3.200, grad=-0.293)]",
            id="right",
        ),
        pytest.param(
            "3 / Value(-3.2)",
            "[Value(-3.200, grad=-0.293), Value(-0.938, grad=1.000), Value(-0.312, grad=3.000), Value(3.000, grad=-0.312)]",
            id="right_neg:1",
        ),
        pytest.param(
            "-3 / Value(-3.2)",
            "[Value(-3.200, grad=0.293), Value(-3.000, grad=-0.312), Value(-0.312, grad=-3.000), Value(0.938, grad=1.000)]",
            id="right_neg:2",
        ),
        pytest.param(
            "Value(323.423) / Value(-2) / Value(-1)",
            "[Value(-161.712, grad=-1.000), Value(-2.000, grad=80.856), Value(-1.000, grad=-161.712), Value(-1.000, grad=161.712), Value(-0.500, grad=-323.423), Value(161.712, grad=1.000), Value(323.423, grad=0.500)]",
            id="two_ops",
        ),
        pytest.param(
            "Value(323.423) / Value(-2) / Value(-1) / 3",
            "[Value(-161.712, grad=-0.333), Value(-2.000, grad=26.952), Value(-1.000, grad=-53.904), Value(-1.000, grad=53.904), Value(-0.500, grad=-107.808), Value(0.333, grad=161.712), Value(3.000, grad=-17.968), Value(53.904, grad=1.000), Value(161.712, grad=0.333), Value(323.423, grad=0.167)]",
            id="three_ops",
        ),
        pytest.param(
            "(3 + (Value(23.1))).f_tanh()",
            "[Value(1.000, grad=1.000), Value(3.000, grad=0.000), Value(23.100, grad=0.000), Value(26.100, grad=0.000)]",
            id="mixed_outside:1",
        ),
        pytest.param(
            "((Value(32) ** 3) / 3323 - Value(5.3)).f_tanh()",
            "[Value(-5.300, grad=0.000), Value(-1.000, grad=0.002), Value(0.000, grad=14.316), Value(1.000, grad=1.000), Value(4.561, grad=0.000), Value(5.300, grad=-0.000), Value(9.861, grad=0.000), Value(32.000, grad=0.000), Value(3323.000, grad=-0.000), Value(32768.000, grad=0.000)]",
            id="mixed_outside:2",
        ),
        pytest.param(
            "((Value(32) * 3) / 3 + Value(5.3)).f_sigmoid()",
            "[Value(0.333, grad=0.000), Value(1.000, grad=1.000), Value(3.000, grad=0.000), Value(3.000, grad=0.000), Value(5.300, grad=0.000), Value(32.000, grad=0.000), Value(32.000, grad=0.000), Value(37.300, grad=0.000), Value(96.000, grad=0.000)]",
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
def test_backward(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backward
    ex.backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
