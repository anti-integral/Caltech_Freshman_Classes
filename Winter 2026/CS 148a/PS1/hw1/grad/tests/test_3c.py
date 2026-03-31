from grad import Value
from grad import nn

import pytest


@pytest.mark.parametrize("N_in", [3, 10, 16])
@pytest.mark.parametrize("N_outs", [[3, 10, 16, 2], [2, 2, 1], [2, 3, 3, 3, 3, 3, 2]])
@pytest.mark.parametrize("activation_fn", ["relu", "sigmoid", "tanh"])
def test_mlp(N_in, N_outs, activation_fn):
    model = nn.MLP(N_in, N_outs, activation_fn)

    x = [1 for _ in range(N_in)]

    # verify forward works
    res = model(x)

    # check dims of outpu
    if isinstance(res, Value):
        assert N_outs[-1] == 1
    else:
        assert len(res) == N_outs[-1]
