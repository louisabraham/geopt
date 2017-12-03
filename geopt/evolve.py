"""
Genetic evolution algorithms
"""

__all__ = ['Evolution']

from copy import deepcopy
from random import randrange, choice, random
from collections import namedtuple

import sympy

from .trees import OpNode, OpTree
from . import operations

# TODO : TEST with levels-back


class Evolution():
    """
    Basic class to create and manage a population
    """

    def __init__(self, inputs, nnodes, outputdim,
                 popsize=5, bestsize=1, newgen=1, ops=None):
        """
        inputs can be int, strings or sympy.Symbol
        """

        if ops is None:
            ops = operations.ops

        if isinstance(inputs, int):
            inputs = ['input_%s' % (i + 1) for i in range(inputs)]
        else:
            inputs = inputs[:]

        for i, var in enumerate(inputs):
            if isinstance(var, str):
                inputs[i] = sympy.symbols(var)

        self.inputs = inputs
        self.nnodes = nnodes
        self.outputdim = outputdim
        self.popsize = popsize
        self.bestsize = bestsize
        self.newgen = newgen
        self.ops = ops

    def random_op(self):
        """
        selects an operation uniformly
        may be changed to use different heuristics
        """
        return choice(self.ops)

    def random_arg(self, inode):
        """
        random argument for OpNode at index inode
        """
        return randrange(len(self.inputs) + inode)

    def random_node(self, i):
        """creates a new node"""
        operation = self.random_op()
        return OpNode(operation, [self.random_arg(i)
                                  for _ in range(operation.arity)])

    def random_tree(self):
        """Generates a tree randomly

        inputdim: dimension of the input
        nnodes: number of nodes
        outputdim: dimension of the output
        ops: list of (function, arity: integer) operations

        returns: a random OpTree
        """
        return OpTree(self.inputs,
                      list(map(self.random_node, range(self.nnodes))),
                      [randrange(len(self.inputs) + self.nnodes)
                       for _ in range(self.outputdim)])

    def new_population(self):
        """
        Initial population
        """
        return [self.random_tree() for _ in range(self.popsize)]

    def random_pick_gene(self, tree):
        """
        chooses a gene uniformly with encoding accepted by mutate_gene
        """
        i = randrange(self.nnodes + self.outputdim)
        if i < self.nnodes:
            return (i, randrange(1 + tree.nodes[i].fun.arity))
        else:
            return i - self.nnodes

    def enumerate_genes(self, tree):
        """
        enumerates all genes with encoding accepted by mutate_gene
        """
        for i in range(self.nnodes):
            for j in range(tree.nodes[i].fun.arity):
                yield (i, j + 1)
            # if you put 0 first then the arity
            # might change at the next step
            yield (i, 0)
        yield from range(self.outputdim)

    def mutate_gene(self, tree, geneid):
        """
        mutates a gene, see the code to understand the encoding
        """
        if isinstance(geneid, int):
            # output gene
            tree.outputs[geneid] = randrange(len(self.inputs) + self.nnodes)
        else:
            inode, igene = geneid
            if igene == 0:
                # mutate the operation and the whole node
                tree.nodes[inode] = self.random_node(inode)
            else:
                # changes just one argument
                tree.nodes[inode] = deepcopy(tree.nodes[inode])
                tree.nodes[inode].args[igene - 1] = self.random_arg(inode)

    def mutate(self, tree, mutation_param):
        """
        point mutation
        if mutation_param is None:
            uses Single
        else:
            uses Accumulate

        returns: new tree evolved from tree
        """
        def is_gene_active(gene, array=tree.active_nodes):
            if isinstance(gene, int):
                return True
            return array[gene[0]]

        tree = deepcopy(tree, {})

        if mutation_param is None:
            while True:
                gene = self.random_pick_gene(tree)
                self.mutate_gene(tree, gene)
                if is_gene_active(gene):
                    break
        else:
            active_node_mutated = False
            while not active_node_mutated:
                for gene in self.enumerate_genes(tree):
                    self.mutate_gene(tree, gene)
                    if is_gene_active(gene):
                        active_node_mutated = True
        return tree

    def evolve(self, population, fit, mutation_param, pool=False, reorder=True):
        """
        population: list of OpTree
        fit: fitness function, greater is better
        pool: False or object with map function (like multiprocessing.Pool)
        reorder: applies reorder on the OpTrees

        Using a pool speeds up the computations, but the fit function
        must be importable in this part of code (it cannot be defined
        in the interpreter or notebook, but wrapping an existing function
        with functools.partial seems to work).


        returns: new population
        """
        # reverse to discard the first more easily
        population = population[::-1]
        if pool:
            fitness = {tid: fitscore for tid, fitscore in
                       zip(map(id, population), pool.map(fit, population))}
            population.sort(key=lambda t: fitness[id(t)], reverse=True)
        else:
            population.sort(key=fit, reverse=True)
        ans = population[:self.bestsize]
        for i in range(self.bestsize, self.popsize - self.newgen):
            ancestor = ans[i % self.bestsize]
            ancestor.reorder()
            ans.append(self.mutate(ancestor, mutation_param))
        for _ in range(self.newgen):
            ans.append(self.random_tree())
        return ans

    def evolvepop(self, gen, pop, fit, mutation_param, pool=False, reorder=True):
        for _ in range(gen):
            pop = self.evolve(pop, fit, mutation_param, pool, reorder)
        return pop

    def evolvenew(self, gen, fit, mutation_param, pool=False, reorder=True):
        """
        returns a new population
        after gen generations
        """
        pop = self.new_population()
        pop = self.evolvepop(gen, pop, fit, mutation_param,
                             pool=False, reorder=True)
        return pop

    def bestpop(self, n_pops, gen, fit, mutation_param, pool=False, reorder=True, range=range):
        """
        returns the best individuals out of n_pops results of evolvenew
        """
        return [max(self.evolvenew(gen, fit, mutation_param, pool, reorder), key=fit)
                for _ in range(n_pops)]
