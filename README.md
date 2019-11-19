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

The algorithm used is described in the draft paper [*Seasonally-Adjusted Auto-Regression of Vector Time Series*, E. Busseti, November 2019.](https://arxiv.org/abs/1911.01010)"


### Installation

**Note: `tsar` is currently under development and is not suitable for
use in production.**

To install, execute in a terminal

```
pip install tsar
```




