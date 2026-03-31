from grad import Value
from grad.lib.topo import topo_sort
import pytest


@pytest.mark.parametrize("x", [1.0, 3.14, 32])
def test_init(x):
    # test if initializing the value is done correctly
    val = Value(x)
    assert val.data == float(x)


@pytest.mark.parametrize("x", [1.0, 3.14, 32])
def test_repr(x):
    # just make sure getting repr str doesn't fail
    _ = str(Value(x))
    assert True


@pytest.mark.parametrize(
    "exp, reference",
    [
        pytest.param(
            "Value(1.0) + Value(12.3)",
            "[Value(1.000, grad=1.000), Value(12.300, grad=1.000), Value(13.300, grad=1.000)]",
            id="simple",
        ),
        pytest.param(
            "Value(-134.0) + Value(12.433)",
            "[Value(-134.000, grad=1.000), Value(-121.567, grad=1.000), Value(12.433, grad=1.000)]",
            id="add_neg",
        ),
        pytest.param(
            "Value(-0.0) + Value(12)",
            "[Value(-0.000, grad=1.000), Value(12.000, grad=1.000), Value(12.000, grad=1.000)]",
            id="zero",
        ),
        pytest.param(
            "Value(3) + Value(-3)",
            "[Value(-3.000, grad=1.000), Value(0.000, grad=1.000), Value(3.000, grad=1.000)]",
            id="opposite",
        ),
        pytest.param(
            "3 + Value(-7.4)",
            "[Value(-7.400, grad=1.000), Value(-4.400, grad=1.000), Value(3.000, grad=1.000)]",
            id="right_neg:1",
        ),
        pytest.param(
            "2 + Value(-3.2)",
            "[Value(-3.200, grad=1.000), Value(-1.200, grad=1.000), Value(2.000, grad=1.000)]",
            id="right_neg:2",
        ),
        pytest.param(
            "Value(15623.32) + 3.2",
            "[Value(3.200, grad=1.000), Value(15623.320, grad=1.000), Value(15626.520, grad=1.000)]",
            id="larger_num",
        ),
        pytest.param(
            "Value(15623.32) + 9.2 + Value(3.321)",
            "[Value(3.321, grad=1.000), Value(9.200, grad=1.000), Value(15623.320, grad=1.000), Value(15632.520, grad=1.000), Value(15635.841, grad=1.000)]",
            id="3sum",
        ),
        pytest.param(
            "Value(15623.32) + 9.2 + Value(3.321) + (Value(-3) + Value(3.321))",
            "[Value(-3.000, grad=1.000), Value(0.321, grad=1.000), Value(3.321, grad=1.000), Value(3.321, grad=1.000), Value(9.200, grad=1.000), Value(15623.320, grad=1.000), Value(15632.520, grad=1.000), Value(15635.841, grad=1.000), Value(15636.162, grad=1.000)]",
            id="4sum",
        ),
    ],
)
def test_add_val(exp, reference):
    # str -> exp
    ex = eval(exp)
    values = topo_sort(ex)

    # backprop
    ex.grad = 1.0
    for val in reversed(values):
        val._backward()

    student_str = str(sorted(values, key=lambda v: (v.data, v.grad)))
    assert student_str == reference
