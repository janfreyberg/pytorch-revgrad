import copy
import pytest
import torch
from pytorch_revgrad import RevGrad


def test_gradients_inverted():
    network = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 1))
    revnetwork = torch.nn.Sequential(copy.deepcopy(network), RevGrad())

    inp = torch.randn(8, 5)
    outp = torch.randn(8, 1)

    criterion = torch.nn.MSELoss()

    criterion(network(inp), outp).backward()
    criterion(revnetwork(inp), outp).backward()

    assert all(
        (p1.grad == -p2.grad).all()
        for p1, p2 in zip(network.parameters(), revnetwork.parameters())
    )


@pytest.mark.parametrize(
    ("alpha_parameter"), [(0.5), (0.7), (0.1), (30.)],
)
def test_gradients_inverted_alpha(alpha_parameter):
    network = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 1))
    revnetwork = torch.nn.Sequential(
        copy.deepcopy(network), RevGrad(alpha=alpha_parameter)
    )

    inp = torch.randn(8, 5)
    outp = torch.randn(8, 1)

    criterion = torch.nn.MSELoss()

    criterion(network(inp), outp).backward()
    criterion(revnetwork(inp), outp).backward()

    for p1, p2 in zip(network.parameters(), revnetwork.parameters()):
        assert torch.isclose(p1.grad, -p2.grad/alpha_parameter).all()
