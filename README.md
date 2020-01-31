# `pytorch-revgrad`

[![PyPI version](https://badge.fury.io/py/pytorch-revgrad.svg)](https://badge.fury.io/py/pytorch-revgrad)
[![Coverage Status](https://coveralls.io/repos/github/janfreyberg/pytorch-revgrad/badge.svg?branch=master)](https://coveralls.io/github/janfreyberg/pytorch-revgrad?branch=master)
[![ci status](https://travis-ci.org/janfreyberg/pytorch-revgrad.svg?branch=master)](https://travis-ci.com/janfreyberg/pytorch-revgrad)
![python version](https://camo.githubusercontent.com/0a3ef56c3f80aca9bc6bafeab605803d81fe2284/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e352532422d626c75652e737667)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package implements a gradient reversal layer for pytorch modules.

## Example usage

```python
import torch

from pytorch_revgrad import RevGrad

model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.Linear(5, 2),
    RevGrad()
)
```
