import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import fedtally as ft  # noqa: E402
from fedtally.pytorch import accuracy, load_params, params_of, trainer  # noqa: E402


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )


def data(site, n=200):
    rng = torch.Generator().manual_seed(site)
    x = torch.randn(n, 4, generator=rng)
    return x, (x[:, 0] + x[:, 1] > 0).long()


def test_params_round_trip_and_leave_buffers_local():
    model = make_model()
    params = params_of(model)
    assert "1.num_batches_tracked" not in params
    assert "1.running_mean" not in params
    assert "1.running_var" not in params
    assert "1.weight" in params
    assert all(v.dtype == np.float32 for v in params.values())
    other = make_model()
    load_params(other, params)
    for name, value in params_of(other).items():
        assert np.array_equal(value, params[name])


def initial():
    torch.manual_seed(0)
    return params_of(make_model())


def test_a_federated_run_trains_a_real_model(tmp_path):
    fed = ft.LocalFederation(
        tmp_path, initial(), sites=3, rounds=5, chunk_coords=64, max_weight=1000
    )
    result = fed.run(trainer(make_model, data, epochs=2, lr=0.1))
    assert result.ok, result.sites
    test_x, test_y = data(99, 500)
    assert accuracy(make_model, result.model, test_x, test_y) > 0.9
    assert len({s["final_digest"] for s in result.sites}) == 1


def test_the_proximal_term_keeps_a_site_nearer_its_start():
    start = initial()

    def moved(mu):
        train = trainer(make_model, data, epochs=5, lr=0.1, proximal_mu=mu)
        out, _ = train(0, 1, start)
        return sum(np.linalg.norm(out[k] - start[k]) for k in start)

    assert moved(10.0) < moved(0.0)


def test_adam_on_the_server_trains_through_a_real_run(tmp_path):
    fed = ft.LocalFederation(
        tmp_path,
        initial(),
        sites=3,
        rounds=5,
        chunk_coords=64,
        max_weight=1000,
        server_adam={"rate": 0.05, "beta1": 0.9, "beta2": 0.99, "tau": 0.001},
    )
    result = fed.run(trainer(make_model, data, epochs=1, lr=0.05))
    assert result.ok, result.sites
    assert result.report["complete"]
    test_x, test_y = data(99, 500)
    assert accuracy(make_model, result.model, test_x, test_y) > 0.8


def test_a_site_trains_the_same_twice():
    start = initial()
    train = trainer(make_model, data, epochs=1, lr=0.1, seed=7)
    first, _ = train(0, 1, start)
    second, _ = train(0, 1, start)
    for name in start:
        assert np.array_equal(first[name], second[name])


class WithRandomBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("projection", torch.randn(4, 4))
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x @ self.projection)


def test_sites_on_threads_start_their_buffers_alike():
    seen = []

    def run(site):
        built = []

        def make():
            built.append(WithRandomBuffer())
            return built[-1]

        train = trainer(make, data, epochs=1, lr=0.1)
        train(site, 1, params_of(built[0]))
        seen.append(built[0].projection.clone())

    threads = [threading.Thread(target=run, args=(site,)) for site in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 4
    assert all(torch.equal(seen[0], other) for other in seen)


def test_frozen_weights_stay_out_of_the_plan():
    model = make_model()
    model[0].weight.requires_grad = False
    params = params_of(model)
    assert "0.weight" not in params
    assert "0.bias" in params


def test_a_parameter_a_plan_cannot_hold_is_refused():
    model = torch.nn.Linear(2, 2, dtype=torch.complex64)
    with pytest.raises(ValueError, match="complex"):
        params_of(model)


def test_batches_of_one_train_a_model_without_batchnorm():
    def plain():
        return torch.nn.Linear(4, 2)

    start = params_of(plain())
    out, _ = trainer(plain, lambda s: data(s, 8), batch_size=1, lr=0.1)(0, 1, start)
    assert not np.array_equal(out["weight"], start["weight"])


def test_batches_of_one_are_refused_for_batchnorm():
    with pytest.raises(ValueError, match="BatchNorm"):
        trainer(make_model, data, batch_size=1)


def test_batchnorm_statistics_can_come_from_other_data():
    x, y = data(99, 500)
    params = initial()
    shifted = x + 5
    a = accuracy(make_model, params, x, y)
    b = accuracy(make_model, params, x, y, calibrate_on=shifted)
    assert a != b


def test_a_remainder_of_one_does_not_stop_a_batchnorm_model():
    train = trainer(make_model, lambda s: data(s, 33), batch_size=32)
    _, weight = train(0, 1, initial())
    assert weight == 33


@pytest.mark.parametrize("examples", [0, 1001])
def test_a_site_the_plan_cannot_weigh_fails_the_run(tmp_path, examples):
    def uneven(site):
        return data(site, examples if site == 0 else 64)

    fed = ft.LocalFederation(
        tmp_path,
        initial(),
        sites=3,
        rounds=1,
        chunk_coords=64,
        max_weight=1000,
        min_cohort=2,
        collect_secs=3,
    )
    result = fed.run(trainer(make_model, uneven, epochs=1, batch_size=64))
    assert not result.ok
    assert not result.sites[0]["ok"]
