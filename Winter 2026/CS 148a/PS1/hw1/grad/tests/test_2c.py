from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.23) ** 23.42",
            "[Value(1.230, grad=2428.060), Value(127.520, grad=1.000)]",
            id="simple",
        ),
        pytest.param(
            "Value(-1.0) ** 3",
            "[Value(-1.000, grad=1.000), Value(-1.000, grad=3.000)]",
            id="neg:1",
        ),
        pytest.param(
            "Value(-43.0) ** (-3)",
            "[Value(-43.000, grad=-0.000), Value(-0.000, grad=1.000)]",
            id="neg:2",
        ),
        pytest.param(
            "(Value(32) ** 3) - 3 * Value(5.3) + 31.432",
            "[Value(-15.900, grad=1.000), Value(-1.000, grad=15.900), Value(3.000, grad=-5.300), Value(5.300, grad=-3.000), Value(15.900, grad=-1.000), Value(31.432, grad=1.000), Value(32.000, grad=3072.000), Value(32752.100, grad=1.000), Value(32768.000, grad=1.000), Value(32783.532, grad=1.000)]",
            id="mixed_ops",
        ),
    ],
)
def test_pow(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
