from grad import Value

def test_loss_decreases():
    w1: Value = Value(1)
    w2: Value = Value(2.3)
    w3: Value = Value(5.7)
    L = w3 * (w1 + w2)
    L.backward()

    eta = 0.1
    for w in [w1, w2, w3]:
        w.data = w.data - w.grad * eta
    L_after = w3 * (w1 + w2)

    print(L.data, L_after.data)
    assert L_after.data < L.data
