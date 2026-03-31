from grad import Value
from grad import nn


import pytest

@pytest.mark.parametrize("N_in", [3, 10, 16])
@pytest.mark.parametrize("N_out", [3, 10, 16, 2])
@pytest.mark.parametrize("activation_fn", ["relu", "sigmoid", "tanh"])
def test_layer(N_in, N_out, activation_fn):
    l = nn.Layer(N_in, N_out, activation_fn)

    x = [1 for _ in range(N_in)]

    # verify forward works
    res = l(x)

    # check internals
    assert l.N_in == N_in
    assert l.N_out == N_out
    assert l.perceptrons[0].activation_fn == activation_fn
    assert len(l.perceptrons) == N_out
    assert l.perceptrons[0].N == N_in

    assert l.nparams() == N_out * (N_in + 1)
