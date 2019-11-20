# `tsar`: time series auto-regressor
<p align="center">
    <a href="https://pypi.org/project/tsar/" alt="tsar on PyPI">
        <img src="https://img.shields.io/pypi/v/tsar.svg" /></a>
    <a href="https://travis-ci.org/enzobusseti/tsar" alt="tsar on TravisCI">
        <img src="https://travis-ci.org/enzobusseti/tsar.svg?branch=master" /></a>
    <a href="https://tsar.readthedocs.io/" alt="tsar's documentation on Read the Docs">
        <img src="https://readthedocs.org/projects/tsar/badge/?version=latest" /></a>
    <a href="https://github.com/enzobusseti/tsar/blob/master/LICENSE" alt="GPLv3 License badge">
        <img src="https://img.shields.io/badge/License-GPLv3-green.svg" /></a>  
</p>

`tsar` is a Python library to model and forecast time series data. It operates on `pandas` dataframes and uses `numba` just-in-time compilation
to speed up some operations.

The core algorithm used is described in the draft paper [*Seasonally-Adjusted Auto-Regression of Vector Time Series*, E. Busseti, November 2019.](https://arxiv.org/abs/1911.01010)


### Installation

**Note: `tsar` is currently in *beta* release, its interface might change.**

To install, execute in a terminal

```
pip install tsar
```

To use it, you need some time series data as
a `pandas` dataframe `data` with date-time index and constant time spacing 
(*i.e.*, its `data.index.freq` attribute must not be `None`).
Only numerical data is supported at this time.

### Build a model
You build a model by 
```
from tsar import tsar
model = tsar(data=data, P=P, F=F)
```
where `P` and `F` are positive integers representing 
how many points in the past will be used for inference, 
and how many steps in the future will be predicted, respectively.

If you wish, you can pass to the constructor any hyper-parameter of the model, for example,
the rank `R` and the quadratic regularization parameter `quadratic_regularization`. 
If not specified, these will be optimized by greedy grid search (see the [paper](https://arxiv.org/abs/1911.01010))
on an internally split test set.

#### Missing data
The data can have any amount of missing values (`np.nan`).

### Infer
Inference is performed with
```
prediction = model.predict(data=new_data, prediction_time=t)
```
where `new_data` is a dataframe with the same column names and time spacing than the one
used to build the model, and prediction_time is the timestamp at which to infer.
The resulting prediction dataframe has time index spanning
from the time `t` minus `P` steps in the past, to the time `t` plus `F` steps in the future.

Any data that was present (and not equal to `np.nan`) in the `new_data` dataframe will be copied to the
appropriate position in the `prediction` dataframe. All other points will be inferred.

#### Missing data
The provided data can have any amount of missing values, the prediction has none.




