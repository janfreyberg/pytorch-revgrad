.. pytorch-swag documentation master file, created by
   sphinx-quickstart on Tue Mar  5 17:18:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

# `pytorch-revgrad` documentation

This package implements a gradient reversal layer for pytorch modules.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   optimizer

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

## Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
