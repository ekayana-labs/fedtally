import json
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import load_file

import fedtally as ft


def model():
    return {
        "w": np.zeros((50, 4), dtype=np.float32),
        "b": np.ones(6, dtype=np.float64),
    }


def toward(values, weights=None):
    """Each site moves everything to its own value, so the mean is known."""

    def train(site, round_number, params):
        weight = 1 if weights is None else weights[site]
        return {k: np.full_like(v, values[site]) for k, v in params.items()}, weight

    return train


def fed(tmp_path, **kwargs):
    kwargs.setdefault("sites", 3)
    kwargs.setdefault("chunk_coords", 32)
    return ft.LocalFederation(tmp_path, model(), **kwargs)


@pytest.mark.parametrize("helpers", [0, 1, 2])
def test_weighted_mean_reaches_every_site(tmp_path, helpers):
    result = fed(tmp_path, helpers=helpers, rounds=2).run(
        toward([0.0, 1.0, 2.0], weights=[1, 1, 2])
    )
    assert result.ok, result.sites
    assert np.allclose(result.model["w"], 1.25, atol=1e-5)
    assert np.allclose(result.model["b"], 1.25, atol=1e-5)
    assert len({s["final_digest"] for s in result.sites}) == 1
    assert result.report["complete"]


def test_model_file_is_ordinary_safetensors(tmp_path):
    result = fed(tmp_path).run(toward([1.0, 1.0, 1.0]))
    loaded = load_file(result.model_path)
    assert loaded["w"].dtype == np.float32
    assert loaded["w"].shape == (50, 4)
    assert np.allclose(loaded["w"], result.model["w"])


def test_training_sees_the_model_from_the_round_before(tmp_path):
    seen = []

    def train(site, round_number, params):
        if site == 0:
            seen.append(float(np.asarray(params["w"]).mean()))
        return {k: np.asarray(v) + 1 for k, v in params.items()}, 1

    fed(tmp_path, rounds=3).run(train)
    assert np.allclose(seen, [0.0, 1.0, 2.0], atol=1e-5)


def test_ml_dsa_identities_work_end_to_end(tmp_path):
    result = fed(tmp_path, sig_alg="ml-dsa-65").run(toward([1.0, 2.0, 3.0]))
    assert result.ok, result.sites
    assert result.report["complete"]


def test_verify_checks_the_model_file(tmp_path):
    result = fed(tmp_path).run(toward([1.0, 2.0, 3.0]))
    assert ft.verify(result.log_path, result.model_path)["complete"]
    other = tmp_path / "other.safetensors"
    ft.save_model(other, model())
    with pytest.raises(ft.VerifyFailed):
        ft.verify(result.log_path, other)


def test_verify_catches_an_edited_log(tmp_path):
    result = fed(tmp_path).run(toward([1.0, 2.0, 3.0]))
    lines = open(result.log_path, "rb").read().splitlines(keepends=True)
    for i in (0, 5, len(lines) - 1):
        edited = list(lines)
        edited[i] = edited[i].replace(b'"kind"', b'"kinD"')
        path = tmp_path / f"edited-{i}.jsonl"
        path.write_bytes(b"".join(edited))
        with pytest.raises(ft.VerifyFailed):
            ft.verify(path)


def test_verify_catches_a_removed_contribution(tmp_path):
    result = fed(tmp_path).run(toward([1.0, 2.0, 3.0]))
    lines = open(result.log_path, "rb").read().splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if b'"kind":"contribute"' in line)
    path = tmp_path / "removed.jsonl"
    path.write_bytes(b"".join(lines[:first] + lines[first + 1:]))
    with pytest.raises(ft.VerifyFailed):
        ft.verify(path)


def test_a_site_refuses_a_plan_for_another_purpose(tmp_path):
    result = fed(tmp_path, purpose="marketing", granted_purpose="cardiology", join_secs=1).run(
        toward([1.0, 2.0, 3.0])
    )
    assert not any(s["ok"] for s in result.sites)
    assert all("refused" in s["error"] for s in result.sites)


def test_a_failing_trainer_is_reported(tmp_path):
    def train(site, round_number, params):
        if site == 2:
            raise RuntimeError("out of memory")
        return {k: np.full_like(v, 1.0) for k, v in params.items()}, 1

    result = fed(tmp_path, collect_secs=3, min_cohort=2).run(train)
    assert not result.sites[2]["ok"]
    assert "out of memory" in result.sites[2]["error"]
    assert result.sites[0]["ok"]
    assert len(result.report["rounds"][0]["included"]) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"w": np.zeros((50, 4), dtype=np.float32)},
        {"w": np.zeros((4, 50), dtype=np.float32), "b": np.zeros(6)},
    ],
)
def test_wrong_tensors_are_reported_not_sent(tmp_path, bad):
    def train(site, round_number, params):
        if site == 0:
            return bad, 1
        return {k: np.full_like(v, 1.0) for k, v in params.items()}, 1

    result = fed(tmp_path, collect_secs=3, min_cohort=2).run(train)
    assert not result.sites[0]["ok"]
    assert result.sites[1]["ok"]


def test_clipped_and_nan_values_do_not_poison_the_sum(tmp_path):
    def train(site, round_number, params):
        out = {k: np.full_like(v, 1.0) for k, v in params.items()}
        if site == 0:
            out["w"] = out["w"].copy()
            out["w"][0, 0] = np.nan
            out["w"][0, 1] = 1e30
        return out, 1

    result = fed(tmp_path, clip=8.0).run(train)
    assert result.ok, result.sites
    assert np.isclose(result.model["w"][0, 0], 2 / 3, atol=1e-5)
    assert np.isclose(result.model["w"][0, 1], 10 / 3, atol=1e-5)
    assert np.isclose(result.model["w"][1, 1], 1.0, atol=1e-5)


def test_separate_processes_with_the_cli(tmp_path):
    import os
    import shutil
    import socket
    import time

    binary = shutil.which("fedtally") or os.environ.get("FEDTALLY_BIN")
    if not binary:
        pytest.skip("the fedtally binary is not on PATH")

    federation = fed(tmp_path, helpers=1)
    federation.save()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        addr = f"127.0.0.1:{s.getsockname()[1]}"

    def spawn(*args):
        return subprocess.Popen([binary, *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    aggregator = spawn(
        "aggregator", "--key", tmp_path / "aggregator.key", "--plan", tmp_path / "plan.jsonl",
        "--model", tmp_path / "initial.safetensors", "--dir", tmp_path / "aggregator", "--listen", addr,
    )
    helper = spawn(
        "helper", "--key", tmp_path / "helper-0.key", "--plan", tmp_path / "plan.jsonl",
        "--dir", tmp_path / "helper-0", "--connect", addr,
    )
    time.sleep(0.2)

    script = """
import json, sys
import numpy as np
import fedtally as ft
i, root, addr, layout = int(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
out = ft.run_site(
    key=f"{root}/site-{i}.key", plan=f"{root}/plan.jsonl", grant=f"{root}/site-{i}.grant.json",
    connect=addr, dir=f"{root}/site-{i}", layout=[tuple(t) for t in layout],
    train=lambda r, p: ({k: np.full_like(v, float(i)) for k, v in p.items()}, 1),
)
print(json.dumps(out))
"""
    sites = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(i), str(tmp_path), addr, json.dumps(federation.layout)],
            stdout=subprocess.PIPE,
        )
        for i in range(3)
    ]
    results = [json.loads(p.communicate(timeout=60)[0]) for p in sites]
    assert all(r["ok"] for r in results), results
    assert aggregator.wait(timeout=60) == 0
    assert helper.wait(timeout=60) == 0
    report = ft.verify(tmp_path / "aggregator" / "log.jsonl", results[0]["final_model"])
    assert report["complete"]
    assert np.allclose(ft.load_model(results[0]["final_model"])["w"], 1.0, atol=1e-5)


def test_a_private_run_reports_what_it_spent(tmp_path):
    privacy = ft.Privacy(epsilon=4.0, delta=1e-5, clip=1.0)
    result = fed(tmp_path, sites=4, rounds=2, privacy=privacy).run(
        toward([1.0, 1.0, 1.0, 1.0], weights=[1, 5, 9, 2])
    )
    assert result.ok, result.sites
    assert result.report["complete"]
    spent = result.report["privacy"]
    assert 0 < spent["epsilon_spent"] <= 4.0 + 1e-6
    assert spent["rho_spent"] <= spent["rho_budget"]
    assert spent["delta"] == 1e-5
    assert all(r["total_weight"] == 4 for r in result.report["rounds"])
    # The clip keeps each site's move to a norm of one, and the rest is noise.
    assert np.abs(result.model["w"]).max() < 5.0
    assert np.asarray(result.model["w"]).std() > 0.01


def test_a_run_without_privacy_says_so(tmp_path):
    result = fed(tmp_path).run(toward([1.0, 1.0, 1.0]))
    assert result.report["privacy"] is None


def test_a_plan_that_may_retry_needs_noise(tmp_path):
    with pytest.raises(ft.FedtallyError, match="no noise"):
        fed(tmp_path / "open", sites=4, min_cohort=2, retries=1)
    privacy = ft.Privacy(epsilon=8.0, delta=1e-5, clip=1.0)
    result = fed(
        tmp_path / "private", sites=4, min_cohort=2, retries=1, privacy=privacy
    ).run(toward([1.0, 1.0, 1.0, 1.0]))
    assert result.ok, result.sites
    assert len(result.report["rounds"]) == 1
    assert not result.report["rounds"][0]["voided"]


def test_a_plan_that_can_lose_a_helper_runs_normally_if_it_loses_none(tmp_path):
    with pytest.raises(ft.FedtallyError, match="min_helpers"):
        fed(tmp_path / "floor", helpers=2, min_helpers=2)
    result = fed(tmp_path / "spare", helpers=3, min_helpers=2).run(
        toward([1.0, 1.0, 1.0])
    )
    assert result.ok, result.sites
    assert result.report["removed_helpers"] == []
    assert len(result.report["rounds"]) == 1


def test_an_impossible_privacy_target_is_refused(tmp_path):
    with pytest.raises(ft.FedtallyError):
        fed(tmp_path, privacy=ft.Privacy(epsilon=1.0, delta=1e-5, clip=-1.0))
    with pytest.raises(ValueError):
        fed(tmp_path, no_such_option=1)


def test_credentials_gate_a_run_and_a_revocation_shows_in_the_report(tmp_path):
    federation = fed(tmp_path, rounds=2, credentials=True, min_cohort=2)

    def train(site, round_number, params):
        if site == 0 and round_number == 1:
            federation.revoke_access(2)
        return {k: np.full_like(v, 1.0) for k, v in params.items()}, 1

    result = federation.run(train)
    assert result.ok, result.sites
    report = result.report
    assert report["complete"]
    kinds = [c["kind"] for c in report["joined"][0]["credentials"]]
    assert "ControlledAccessGrants" in kinds and "Provenance" in kinds
    assert report["joined"][0]["dataset"].startswith("did:key:")
    assert [len(r["included"]) for r in report["rounds"]] == [3, 2]
    assert len(report["revoked"]) == 1
    assert "revoked" in report["revoked"][0]["reason"]


def test_sampling_draws_some_sites_each_round(tmp_path):
    result = fed(tmp_path, sites=4, rounds=3, sample=2, min_cohort=2).run(
        toward([1.0, 1.0, 1.0, 1.0])
    )
    assert result.ok, result.sites
    assert result.report["complete"]
    for round_report in result.report["rounds"]:
        assert len(round_report["invited"]) == 2
        assert len(round_report["included"]) == 2


def test_parties_can_go_by_did_bio(tmp_path):
    result = fed(tmp_path, did_bio="devnet").run(toward([1.0, 1.0, 1.0]))
    assert result.ok, result.sites
    assert result.report["aggregator"].startswith("did:bio:devnet:")
    assert all(j["site"].startswith("did:bio:devnet:") for j in result.report["joined"])


def test_a_site_past_the_norm_bound_is_left_out(tmp_path):
    privacy = ft.Privacy(epsilon=8.0, delta=1e-5, clip=1.0)
    pull = toward([0.0, 0.0, 0.0, 1e6])
    honest = fed(tmp_path / "a", sites=4, privacy=privacy, bounded=True, cheats={3: 4096})
    result = honest.run(pull)
    assert all(result.sites[i]["ok"] for i in range(3)), result.sites
    round_ = result.report["rounds"][0]
    assert round_["total_weight"] == 3
    assert len(round_["excluded"]) == 1
    assert np.abs(result.model["w"]).max() < 5.0

    open_ = fed(tmp_path / "b", sites=4, privacy=privacy, cheats={3: 4096}).run(pull)
    assert open_.report["rounds"][0]["total_weight"] == 4
    assert np.abs(open_.model["w"]).max() > 50.0


def test_scoring_reports_groups_and_rewards(tmp_path):
    result = fed(tmp_path, sites=4, rounds=2, scoring=2).run(
        toward([2.0, 2.0, 2.0, -2.0])
    )
    assert result.ok, result.sites
    for round_ in result.report["rounds"]:
        assert len(round_["groups"]) == 2
        assert sum(g["share"] for g in round_["groups"]) == 1_000_000
    rewards = result.report["rewards"]
    assert len(rewards) == 4
    assert abs(sum(r["share"] for r in rewards) - 1.0) < 1e-9
