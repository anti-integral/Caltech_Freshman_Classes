"""
tiny Neural Network library built on top of your autograd implementation.
"""

from grad import Value
import random  # use random.uniform to generate random weights and biases


class Module(object):
    def nparams(self) -> int:
        return len(self.params())

    def params(self) -> list[Value]:
        return []

    def step(self, lr) -> None:
        # iterate through every parameter, and then update it
        for p in self.params():
            p = p - lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params():
            p.zero_grad()

    def __repr__(self) -> str:
        return super().__repr__()


class Perceptron(Module):
    """
    TODO: implement a perceptron that takes in an input vector in R^n and outputs a single value
    Perceptron: R^n -> R
    (1) x is a list with length N (representing a vector in R^n)
    (2) signal = x_1*w_1 + x_2*w_2 + ... + x_N*w_N
    (3) activate = activation_fn (s) where s = signal(x)
    NOTE: activation_fn can be "relu" "sigmoid" or "tanh"
    """

    def __init__(self, N: int, activation_fn="relu"):
        self.N = N  # num of inputs to this perceptron
        self.w: list[Value] = (
            [Value(random.uniform(-1, 1)) for _ in range(N)]
        )
        self.b: Value = Value(0.0)  # TODO: initialize bias as 0 (single number)
        self.activation_fn = activation_fn

    def signal(self, x: list[Value]) -> Value:
        s = self.b
        for w_i, x_i in zip(self.w, x):
            s = s + (w_i * x_i)
        return s

    def activate(self, s: Value) -> Value:
        # TODO: given the signal s, activate and return it
        # you'll have to write some if statements to conditionally use f_relu, f_sigmoid, or f_tanh
        if self.activation_fn == "relu":
            return s.f_relu()
        if self.activation_fn == "sigmoid":
            return s.f_sigmoid()
        if self.activation_fn == "tanh":
            return s.f_tanh()
        raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def params(self) -> list[Value]:
        # TODO: return a list of all parameters of the Neuron
        # hint: weights and biases
        return self.w + [self.b]

    def __call__(self, x) -> Value:
        """
        Example usage:
        >>> p = Perceptron(3) # notice N=3
        >>> p([1, 2, 3]) # notice input is a vector in R^3
        """
        if len(x) != self.N:
            raise Exception(f"Expected vector of length: {self.N}")
        return self.activate(self.signal(x))

    def __repr__(self) -> str:
        return f"Neuron({self.N})"


class Layer(Module):
    """
    TODO: implement a perceptron layer
    Layer: R^N_in -> R^N_out
    (1) input x is a list with length N_in (representing a vector in R^N_in)
    (2) a layer has R^N_out perceptrons
    (3) each perceptron spits out a single number in R
    (4) put them all together in an output vector/list of length N_out
    """

    def __init__(self, N_in, N_out, activation_fn: str) -> None:
        self.N_in = N_in  # number of input features
        self.N_out = N_out  # number of output features
        self.perceptrons = [
            Perceptron(N_in, activation_fn) for _ in range(N_out)
        ]  # TODO: initialize a list of Perceptrons in this layer

    def __call__(self, x) -> Value | list[Value]:
        if isinstance(x, Value):
            x = [x]
        assert len(x) == self.N_in
        out: list[Value] = [p(x) for p in self.perceptrons]  # TODO: pass input x to each perceptron
        if len(out) == 1:
            # NOTE: if out is a single number just return it. Ex: [n] -> n
            return out[0]
        else:
            return out

    def params(self) -> list[Value]:
        # TODO: return a list of all params in this Layer, hint: you can reuse Perceptron's params method
        params: list[Value] = []
        for perceptron in self.perceptrons:
            params.extend(perceptron.params())
        return params

    def __repr__(self) -> str:
        return f"Layer(N_in={self.N_in}, N_out={self.N_out})"


class MLP(Module):
    """
    Implement an MLP (multi layer perceptron)
    (1) You are given N_in so x will be a vector in R^N_in
    (2) N_outs is a list of integers (the number of outputs of each layer)
    (3) Example:
        - if N_in=10 and N_outs=[12, 16, 14, 3] then self.N = [10] + [12, 16, 14, 3] = [10, 12, 16, 14, 3]
        - input in R^10
        - layer #1: R^10 -> R^12
        - layer #2: R^12 -> R^16
        - layer #3: R^16 -> R^14
        - layer #4: R^14 -> R^3
    (4) Your job is to initialize these layers in self.layers

    Example Usage
    -------------
    >>> model = MLP(3, [10, 10, 2], "relu")
    >>> model([1, 2, 3])
    """

    def __init__(self, N_in: int, N_outs: list[int], activation_fn: str):
        self.N = [N_in] + N_outs
        # TODO: initialize a list of Layer's, with the correct in/out dims for each one (hint: self.N has been assigned for you)
        self.layers = [
            Layer(self.N[i], self.N[i + 1], activation_fn) for i in range(len(self.N) - 1)
        ]

    def __call__(self, x) -> Value | list[Value]:
        # TODO: pass x through each Layer in self.layers
        for i, layer in enumerate(self.layers):
            if isinstance(x, Value):
                x = [x]
            x = layer(x)
        return x

    def params(self) -> list[Value]:
        # TODO: return a list of all params in this MLP, hint: you can reuse Layer's params method
        params: list[Value] = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def __repr__(self) -> str:
        # prints out mlp as a readable string
        return f"MLP({self.N}): {[str(l) for l in self.layers]}]"
