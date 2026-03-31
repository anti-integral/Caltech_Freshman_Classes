from grad import Value
from grad import nn
import pytest


@pytest.mark.parametrize("N", [3, 10, 16])
@pytest.mark.parametrize("activation_fn", ["relu", "sigmoid", "tanh"])
def test_perceptron(N, activation_fn):
    p = nn.Perceptron(N, activation_fn)

    x = [1 for _ in range(N)]

    # verify forward works
    res = p(x)
    assert isinstance(res, Value)

    # check internal lengths and check if activation fn was set right
    assert len(p.w) == N
    assert activation_fn in p.activation_fn

    # check num_params
    assert p.nparams() == N + 1
