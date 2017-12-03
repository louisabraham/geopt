"""
Operation trees

TODO: add support for tensorflow
"""

__all__ = ['OpNode', 'OpTree']


from collections import namedtuple, defaultdict
import os
from copy import deepcopy
from random import choice

import sympy
import numpy as np
from graphviz import Digraph
# if the like above fails, add the path
# to the Graphviz executable like this
# os.environ["PATH"] += os.pathsep + '/path/to/Graphviz/bin'


from .util import inf_to_nan, make_of_shape, complex_to_nan

OpNode = namedtuple('OpNode', 'fun args')


class OpTree(namedtuple('OpTree', 'inputs nodes outputs')):
    """
    inputs: list of str or None
    nodes: list of OpNode
    outputs: list of indexes
    """

    @property
    def active_nodes(self):
        if not hasattr(self, '_active_nodes'):
            nn = len(self.nodes)
            ni = len(self.inputs)
            Q = self.outputs.copy()
            ans = self._active_nodes = [False] * nn
            while Q:
                x = Q.pop() - ni
                if x >= 0 and not ans[x]:
                    ans[x] = True
                    Q += self.nodes[x].args
        return self._active_nodes

    def _evaluate(self, *values, usenp=False):
        """
        Evaluates the OpTree over the inputs in an efficient way
        (only the required nodes are calculated)

        alwais use evaluate or expr

        returns: list of output values
        """
        ni = len(values)
        assert len(self.inputs) == ni
        nn = len(self.nodes)

        ans = [None] * (ni + nn)
        ans[:ni] = values
        for i in filter(self.active_nodes.__getitem__, range(nn)):
            node = self.nodes[i]
            ans[i + ni] = (node.fun
                           if not usenp
                           else node.fun.npfun
                           )(*map(ans.__getitem__, node.args))

        return list(map(ans.__getitem__, self.outputs))

#    @property
#    def symbols(self):
#        """
#        Symbols of the variables
#        """
#        if not hasattr(self, '_symbols'):
#            self._symbols = sympy.symbols(self.inputs)
#        return self._symbols

    @property
    def expr(self):
        """
        mathematical expression of the outputs
        Replaces sp.zoo with sp.nan
        """
        if not hasattr(self, '_expr'):
            self._expr = [complex_to_nan(e)
                          for e in self._evaluate(*self.inputs)]
        return self._expr

    def lambdify(self, backend):
        """
        Returns a lambda function using sympy

        NEVER TRY to memoise this function
        https://stackoverflow.com/questions/45085095/passing-sympy-lambda-to-multiprocessing-pool-map
        """
        return [sympy.lambdify(self.inputs, e, backend)
                for e in self.expr]

    def evaluate(self, *values, backend=None):
        """
        Numeric evaluation

        if backend is None, uses numpy arrays without simplification.
        You can also use some backends of the sympy.lambdify function,
        aka 'math', 'numpy' and 'numexpr'.
        Note that here numpy will be used inside sympy
        """
        if backend is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                return self._evaluate(*values, usenp=True)
        return [make_of_shape(values[0].shape)
                (inf_to_nan(f(*values))
                 if e != sympy.nan else np.nan)
                for e, f in zip(self.expr, self.lambdify(backend))
                ]

    def reorder(self):
        """
        Reorders the nodes
        """
        ni = len(self.inputs)
        nn = len(self.nodes)

        # feeds_to[i] contains the indexes of the nodes
        # depending on the node i
        feeds_to = defaultdict(list)
        for inode, node in enumerate(self.nodes, ni):
            for source in node.args:
                # feeds_to[source] can contain duplicates
                feeds_to[source].append(inode)

        required_count = {ni + inode: len(self.nodes[inode].args)
                          for inode in range(nn)}
        for source in range(ni):
            for inode in feeds_to[source]:
                required_count[inode] -= 1
        newloc = {i: i for i in range(ni)}
        oldloc = {i: i for i in range(ni)}
        addable = set(inode for inode, count in required_count.items()
                      if count == 0)
        for loc in range(ni, ni + nn):
            inode = choice(tuple(addable))
            addable.remove(inode)
            newloc[inode] = loc
            oldloc[loc] = inode
            fun, args = self.nodes[inode - ni]
            for jnode in feeds_to[inode]:
                required_count[jnode] -= 1
                if required_count[jnode] == 0:
                    addable.add(jnode)
        for _, args in self.nodes:
            for i, arg in enumerate(args):
                args[i] = newloc[arg]
        new_nodes = [None] * nn
        for i in range(nn):
            new_nodes[newloc[ni + i] - ni] = self.nodes[i]
        self.nodes[:] = [self.nodes[oldloc[ni + i] - ni] for i in range(nn)]
        self.outputs[:] = map(newloc.__getitem__, self.outputs)
        if hasattr(self, '_active_nodes'):
            delattr(self, '_active_nodes')

    def visualize(self, light=True):
        """
        GraphViz representation of the graph
        if light=True, only display the active nodes and inputs
        if light=False, displays all nodes

        TODO: look at d-CGP for better display
        """
        ans = Digraph()
        ni = len(self.inputs)
        nn = len(self.nodes)
        used = [False] * (nn + ni)

        # DFS to calculate used
        Q = self.outputs.copy()
        while Q:
            x = Q.pop()
            if not used[x]:
                used[x] = True
                if x >= ni:
                    Q += self.nodes[x - ni].args
        for i in self.outputs:
            used[i] = 2

        for i in range(ni):
            if used[i]:
                ans.node(str(i), str(self.inputs[i]), style='',
                         color='red' if used[i] == 2 else 'b')
            elif not light:
                ans.node(str(i), str(self.inputs[i]), style='dotted')
        for i in range(nn):
            node = self.nodes[i]
            if used[ni + i]:
                ans.node(str(ni + i), node.fun.name, style='',
                         color='red' if used[ni + i] == 2 else 'b')
            elif not light:
                ans.node(str(ni + i), node.fun.name, style='dotted')
            for n, v in enumerate(node.args, 1):
                if not light or used[v] and used[ni + i]:
                    ans.edge(str(v), str(ni + i), '%s' %
                             n if not node.fun.is_commutative else None)
        return ans

    def __deepcopy__(self, d):
        return OpTree(self.inputs, deepcopy(self.nodes), self.outputs[:])
