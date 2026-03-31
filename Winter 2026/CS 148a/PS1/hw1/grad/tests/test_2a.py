from grad import Value
from grad.lib.topo import topo_sort
import pytest
from _pytest.reports import TestReport


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23) * Value(23.42)",
            "[Value(1.230, grad=23.420), Value(23.420, grad=1.230), Value(28.807, grad=1.000)]",
            id="simple",
        ),
        pytest.param(
            "Value(13920423.23) * Value(3242.42)",
            "[Value(3242.420, grad=13920423.230), Value(13920423.230, grad=3242.420), Value(45135858689.417, grad=1.000)]",
            id="simple_big",
        ),
        pytest.param(
            "Value(-1.0) * Value(32.23)",
            "[Value(-32.230, grad=1.000), Value(-1.000, grad=32.230), Value(32.230, grad=-1.000)]",
            id="neg:1",
        ),
        pytest.param(
            "Value(-43.0) * Value(-342.254)",
            "[Value(-342.254, grad=-43.000), Value(-43.000, grad=-342.254), Value(14716.922, grad=1.000)]",
            id="neg:2",
        ),
        pytest.param(
            "3 * Value(3.2)",
            "[Value(3.000, grad=3.200), Value(3.200, grad=3.000), Value(9.600, grad=1.000)]",
            id="right",
        ),
        pytest.param(
            "3 * Value(-3.2)",
            "[Value(-9.600, grad=1.000), Value(-3.200, grad=3.000), Value(3.000, grad=-3.200)]",
            id="right_neg:1",
        ),
        pytest.param(
            "-3 * Value(-3.2)",
            "[Value(-3.200, grad=-3.000), Value(-3.000, grad=-3.200), Value(9.600, grad=1.000)]",
            id="right_neg:2",
        ),
        pytest.param(
            "Value(323.423) * Value(-2) * Value(-1)",
            "[Value(-646.846, grad=-1.000), Value(-2.000, grad=-323.423), Value(-1.000, grad=-646.846), Value(323.423, grad=2.000), Value(646.846, grad=1.000)]",
            id="two_ops",
        ),
        pytest.param(
            "Value(323.423) * Value(-2) * Value(-1) * 3",
            "[Value(-646.846, grad=-3.000), Value(-2.000, grad=-970.269), Value(-1.000, grad=-1940.538), Value(3.000, grad=646.846), Value(323.423, grad=6.000), Value(646.846, grad=3.000), Value(1940.538, grad=1.000)]",
            id="three_ops",
        ),
        pytest.param(
            "(Value(32) * 3) * 3 * Value(5.3)",
            "[Value(3.000, grad=508.800), Value(3.000, grad=508.800), Value(5.300, grad=288.000), Value(32.000, grad=47.700), Value(96.000, grad=15.900), Value(288.000, grad=5.300), Value(1526.400, grad=1.000)]",
            id="order_ops",
        ),
        pytest.param(
            "(Value(32) + 3) + 3 * Value(5.3)",
            "[Value(3.000, grad=1.000), Value(3.000, grad=5.300), Value(5.300, grad=3.000), Value(15.900, grad=1.000), Value(32.000, grad=1.000), Value(35.000, grad=1.000), Value(50.900, grad=1.000)]",
            id="mixed_ops",
        ),
    ],
)
def test_mul(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
