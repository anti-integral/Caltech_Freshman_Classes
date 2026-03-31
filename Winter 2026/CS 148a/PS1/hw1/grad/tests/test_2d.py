from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23) / Value(23.42)",
            "[Value(0.043, grad=1.230), Value(0.053, grad=1.000), Value(1.230, grad=0.043), Value(23.420, grad=-0.002)]",
            id="simple",
        ),
        pytest.param(
            "Value(13920423.23) / Value(3242.42)",
            "[Value(0.000, grad=13920423.230), Value(3242.420, grad=-1.324), Value(4293.220, grad=1.000), Value(13920423.230, grad=0.000)]",
            id="simple_big",
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
            "(Value(32) ** 3) / 3 - Value(5.3)",
            "[Value(-5.300, grad=1.000), Value(-1.000, grad=5.300), Value(0.333, grad=32768.000), Value(3.000, grad=-3640.889), Value(5.300, grad=-1.000), Value(32.000, grad=1024.000), Value(10917.367, grad=1.000), Value(10922.667, grad=1.000), Value(32768.000, grad=0.333)]",
            id="mixed_ops:1",
        ),
        pytest.param(
            "(Value(32) * 3) - 3 * ((-1) * Value(3) / Value(5.3))",
            "[Value(-3.000, grad=-0.566), Value(-1.698, grad=-1.000), Value(-1.000, grad=-1.698), Value(-1.000, grad=-1.698), Value(-0.566, grad=-3.000), Value(0.189, grad=9.000), Value(1.698, grad=1.000), Value(3.000, grad=0.566), Value(3.000, grad=0.566), Value(3.000, grad=32.000), Value(5.300, grad=-0.320), Value(32.000, grad=3.000), Value(96.000, grad=1.000), Value(97.698, grad=1.000)]",
            id="mixed_ops:2",
        ),
    ],
)
def test_div(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference


def test_zero_div():
    with pytest.raises(ZeroDivisionError):
        _ = Value(1) / Value(0)
