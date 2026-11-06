"""Training a PyTorch model at each site. Importing this needs torch.

Only trainable parameters travel. A buffer such as BatchNorm's running statistics
isn't a gradient: the server's step would drive a running variance negative, and
in a private run it would eat into the clip. So buffers and frozen weights aren't
sent or kept. Each round every site starts them over from one template built from
`build_seed`, and `accuracy` measures BatchNorm statistics afresh.
"""

import copy
import threading

import numpy as np
import torch

_BUILDING = threading.Lock()
_FLOATS = (torch.float32, torch.float64)


def params_of(model) -> dict:
    """A model's trainable parameters as arrays, by name."""
    params = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if value.dtype not in _FLOATS:
            raise ValueError(f"{name} is {value.dtype}, and a plan holds floats only")
        params[name] = value.detach().cpu().numpy().copy()
    return params


def load_params(model, params: dict) -> None:
    """Loads arrays from `params_of` into a model, leaving anything else alone."""
    state = model.state_dict()
    for name, value in params.items():
        state[name] = torch.as_tensor(np.array(value), dtype=state[name].dtype)
    model.load_state_dict(state)


def build(make_model, seed: int = 0):
    """A model from `seed`, leaving torch's global generator as it found it. Sites
    may run on threads of one process, so building takes turns.
    """
    with _BUILDING, torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return make_model()


def _has_batchnorm(model) -> bool:
    return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def trainer(
    make_model,
    data_for_site,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    momentum: float = 0.0,
    proximal_mu: float = 0.0,
    loss=None,
    device: str = "cpu",
    seed: int = 0,
    build_seed: int = 0,
):
    """A `train(site, round, params)` for `LocalFederation.run`.

    `data_for_site(site)` returns `(inputs, targets)` as tensors. `proximal_mu`
    above zero is FedProx: each step is also pulled back toward the round's
    starting model by `mu / 2 * ||w - w_start||^2`. It changes only what a site
    computes, so the plan and every other party are unaware of it.

    `seed` fixes each site's shuffling. Dropout draws from torch's global
    generator, which sites on threads share, so it is not reproducible there.
    """
    loss = loss or torch.nn.functional.cross_entropy
    template = build(make_model, build_seed)
    if batch_size == 1 and _has_batchnorm(template):
        raise ValueError("BatchNorm can't train on batches of one")

    def train(site, round_number, params):
        model = copy.deepcopy(template).to(device)
        load_params(model, params)
        start = [p.detach().clone() for p in model.parameters()]
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        inputs, targets = data_for_site(site)
        if len(inputs) == 0:
            raise ValueError(f"site {site} has no examples to train on")
        batchnorm = _has_batchnorm(model)
        if len(inputs) == 1 and batchnorm:
            raise ValueError(f"site {site} has one example, which BatchNorm can't train on")
        inputs, targets = inputs.to(device), targets.to(device)
        shuffle = torch.Generator(device=device)
        shuffle.manual_seed(seed + 1009 * site + round_number)
        model.train()
        for _ in range(epochs):
            order = torch.randperm(len(inputs), device=device, generator=shuffle)
            for batch in order.split(batch_size):
                if len(batch) == 1 and batchnorm:
                    continue
                optimiser.zero_grad()
                cost = loss(model(inputs[batch]), targets[batch])
                if proximal_mu > 0:
                    pulled = sum(
                        (p - s).pow(2).sum() for p, s in zip(model.parameters(), start)
                    )
                    cost = cost + proximal_mu / 2 * pulled
                cost.backward()
                optimiser.step()
        return params_of(model), len(inputs)

    return train


@torch.no_grad()
def calibrate(model, inputs) -> None:
    """Sets a model's BatchNorm statistics from `inputs`, as the run doesn't carry
    them. Do this before using a run's model, on data like what it will see.
    """
    norms = [
        m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not norms:
        return
    model.eval()
    for layer in norms:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    model(inputs)
    model.eval()


@torch.no_grad()
def accuracy(
    make_model,
    params: dict,
    inputs,
    targets,
    device: str = "cpu",
    build_seed: int = 0,
    calibrate_on=None,
) -> float:
    """The share of `targets` a model with `params` gets right. BatchNorm layers
    take their statistics from `calibrate_on`, or else from the `inputs` scored,
    which lets the mix being scored affect the score.
    """
    model = build(make_model, build_seed).to(device)
    load_params(model, params)
    inputs, targets = inputs.to(device), targets.to(device)
    calibrate(model, inputs if calibrate_on is None else calibrate_on.to(device))
    model.eval()
    return (model(inputs).argmax(dim=1) == targets).float().mean().item()
