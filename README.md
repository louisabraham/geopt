# GeOpt by GEOpT

aka **Ge**neric **Opt**imization by **G**enetically **E**volved **Op**eration **T**rees

This library allows feature engineering by genetically evolved operation trees
(in fact they are [DAGs](https://en.wikipedia.org/wiki/Directed_acyclic_graph), but `geopd` sounded ugly).

The trees look like this (output in red):

<img src="https://raw.githubusercontent.com/louisabraham/geopt/master/random_tree.png"
alt="random tree" width="70%"/>


## Optimizations

The most two most important optimizations are:

- the reduction of the operation trees by using [SymPy](http://www.sympy.org/en/index.html)
- good numerical performances when using the [numexpr](https://github.com/pydata/numexpr) backend (optional argument of `OpTree.evaluate`).

The `Evolution.evolve` can also take a `pool` argument and be parallelized.


## Performances

The [`binary_classification.ipynb`](https://github.com/louisabraham/geopt/blob/master/examples/binary_classification.ipynb) notebook shows how a simple threshold model
trained on **20%** of the data yields **95.6%** accuracy on the
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)


## Possible uses 

The examples include:

- `binary_classification.ipynb`: simple binary classification with threshold evolved with area under ROC curve
- `symbolic_regression.ipynb`


`geopt` can be used as the first layer of **any** model. For example it could be of great help
as feature engineering in linear classifiers or neural networks.

However, you want the model used in the `fit` function to be quite cheap to evaluate,
even if you use another model in the end.

This is just a proof of concept, the pull requests are very welcome.


## Usage

Look at the `examples/` folder.

Also, the code is really not big (~ 600 SLOC), quite readable and well commented.


## More ideas

### About the model
- Add numeric parameter nodes, compute the derivatives using SymPy and do gradient descent

### About the optimization
- Since many OpTrees are similar in the population and over the generations,
one could memoize their fitness on the current dataset. A simple way to do it would be to make a dictionnary
`{formula: fitness}`.
- One could think this one step further with a node grained memoization `{node: evaluation}`,
with the drawback that it could only work with the 'raw' numpy backend (`backend=None`). However, the memory
footprint would be horrible, and numexpr offers great performances so I don't think it would be a good idea.
- Clustering would be possible since the OpTrees are picklable, with an interface like this
[root client](https://github.com/louisabraham/MarsRover/blob/master/remote_executor.py) and those
[node servers](https://github.com/louisabraham/MarsRover/blob/master/rest.py).