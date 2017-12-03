"""
Defines operations used in trees
"""

__all__ = ['ops', 'ops_simple', 'add', 'sub',
           'mul', 'divide', 'sqrt', 'log', 'square']

import operator

import numpy as np
import sympy as sp


class Op():
    """
    Class to represent operations
    """

    def __init__(self, name, fun, arity, is_commutative=False, npfun=None):
        if npfun is None:
            npfun = fun
        self.name = name
        self.fun = fun
        self.np = npfun
        self.arity = arity
        if self.arity == 1:
            is_commutative = True
        self.is_commutative = is_commutative

    def __call__(self, *args):
        assert len(args) == self.arity
        return self.fun(*args)

    def npfun(self, *args):
        assert len(args) == self.arity
        with np.errstate(divide='ignore', invalid='ignore'):
            ans = self.np(*args)
            return np.where(np.isinf(ans), np.nan, ans)

    def __repr__(self):
        return self.name

    def __deepcopy__(self, d):
        return self


add = Op('+', operator.add, 2, True)
sub = Op('-', operator.sub, 2)
mul = Op('*', operator.mul, 2, True)
divide = Op('/', operator.truediv, 2)
sqrt = Op('sqrt', sp.sqrt, 1, npfun=np.sqrt)
log = Op('log', sp.log, 1, npfun=np.log)


# NEVER EVER try to use a decorator, it makes multiprocessing fail

def _square(x):
    return x * x


square = Op('**2', _square, 1, npfun=np.square)


def _minus(x):
    return -x


minus = Op('-', _minus, 1)


def _th_if(x, y, z):
    return sp.Piecewise((y, x >= 0), (z, True))


def _th_if_np(x, y, z):
    return np.where(x >= 0, y, z)


th_if = Op('if >= 0', _th_if, 3, npfun=_th_if_np)


ops_simple = [
    add,
    sub,
    mul,
    divide,
    minus,
    #    th_if
]


ops = ops_simple + [
    sqrt,
    log,
    square
]
