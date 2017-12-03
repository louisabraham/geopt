import numpy as np
import sympy


def complex_to_nan(ans):
    """
    ans is a sympy expression
    """
    ans = ans.replace(sympy.zoo, sympy.nan)
    return ans if ans.is_real != False else sympy.nan


def inf_to_nan(ans):
    """
    ans is a numpy array
    """
    return np.where(np.isinf(ans), np.nan, ans)


def make_of_shape(shape):
    def f(value):
        if hasattr(value, 'shape') and value.shape == shape:
            return value
        return np.full(shape, value)
    return f


class DummyModel:

    def __init__(self, inputs, expression_string):
        if isinstance(inputs, int):
            inputs = ['input_%s' % (i + 1) for i in range(inputs)]
        else:
            inputs = inputs[:]

        for i, var in enumerate(inputs):
            if isinstance(var, str):
                inputs[i] = sympy.symbols(var)
        self.inputs = inputs

        self.expr = [sympy.S(expression_string)]

    def lambdify(self, backend):
        """
        Returns a lambda function using sympy
        """
        return [sympy.lambdify(self.inputs, e, backend)
                for e in self.expr]

    def evaluate(self, *values, backend=None):
        """
        Numeric evaluation

        if backend is None, uses default backend of sympy
        You can also use any possible backend of the sympy.lambdify function
        """
        fun = sympy.lambdify(self.inputs, self.expr[0], backend)
        return [make_of_shape(values[0].shape)
                (inf_to_nan(fun(*values))
                 if self.expr[0] != sympy.nan else np.nan)]
