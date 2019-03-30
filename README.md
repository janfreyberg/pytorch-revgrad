# `pytorch-revgrad`

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
