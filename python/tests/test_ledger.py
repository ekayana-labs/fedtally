"""An anchored run against a local validator. Skipped without one."""

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import fedtally as ft

PROGRAM = Path(__file__).parents[2] / "ledger/target/deploy/fedtally_ledger.so"
SOL = 1_000_000_000

pytestmark = pytest.mark.skipif(
    shutil.which("solana-test-validator") is None or not PROGRAM.exists(),
    reason="needs solana-test-validator and the built program",
)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def call(url, method, *params):
    request = urllib.request.Request(
        url,
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as answer:
        return json.load(answer)


def program_id():
    source = Path(__file__).parents[2] / "ledger/program/src/lib.rs"
    return re.search(r'pub const ID.*?"(\w+)"', source.read_text(), re.S).group(1)


@pytest.fixture
def validator(tmp_path):
    rpc = free_port()
    child = subprocess.Popen(
        [
            "solana-test-validator",
            "--ledger", str(tmp_path / "chain"),
            "--reset", "--quiet",
            "--bind-address", "127.0.0.1",
            "--rpc-port", str(rpc),
            "--faucet-port", str(free_port()),
            "--mint", "11111111111111111111111111111112",
            "--bpf-program", program_id(), str(PROGRAM),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{rpc}"
    try:
        for _ in range(120):
            try:
                if call(url, "getSlot").get("result", 0) > 0:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        yield url
    finally:
        child.terminate()
        child.wait()


def fund(url, wallet):
    call(url, "requestAirdrop", wallet, 100 * SOL)
    for _ in range(120):
        if call(url, "getBalance", wallet)["result"]["value"] > 0:
            return
        time.sleep(0.25)
    raise AssertionError(f"{wallet} was not funded")


def test_a_run_anchored_from_python_settles_and_returns_every_rent(tmp_path, validator):
    initial = {"w": np.zeros(300, dtype=np.float32)}
    fed = ft.LocalFederation(
        tmp_path / "run", initial, sites=3, rounds=2, chunk_coords=128, did_bio="localnet"
    )
    wallets = fed.wallets()
    for wallet in [wallets["publisher"], wallets["aggregator"], wallets["watcher"], *wallets["sites"]]:
        fund(validator, wallet)

    ledger = ft.Ledger(
        validator,
        ft.Stakes(
            site_stake=SOL,
            aggregator_stake=10 * SOL,
            reward_per_round=3_000_000,
            dispute_bond=SOL,
            challenge_slots=25,
            stall_slots=250,
        ),
        poll_ms=100,
    )

    def train(site, round_number, params):
        return {"w": np.asarray(params["w"]) + 1.0}, 1

    result = fed.run(train, ledger=ledger)
    assert result.ok, (result.sites, result.aggregator_error, result.ledger)
    assert np.allclose(result.model["w"], 2.0, atol=1e-4)
    sent = result.ledger["sent"]
    assert sent["settle"] == 2
    assert sent["claim"] == 2 * 3
    assert sent["round_close"] == 2
    assert sent["withdraw (site)"] == 3
    assert sent["withdraw (aggregator)"] == 1


def test_separate_parties_anchor_with_run_site_and_run_anchor(tmp_path, validator):
    binary = shutil.which("fedtally") or os.environ.get("FEDTALLY_BIN")
    if not binary:
        pytest.skip("the fedtally binary is not on PATH")
    initial = {"w": np.zeros(300, dtype=np.float32)}
    federation = ft.LocalFederation(
        tmp_path, initial, sites=3, rounds=1, chunk_coords=128, did_bio="localnet"
    )
    federation.save()
    wallets = federation.wallets()
    for wallet in [wallets["publisher"], wallets["aggregator"], *wallets["sites"]]:
        fund(validator, wallet)
    addr = f"127.0.0.1:{free_port()}"
    plan = tmp_path / "plan.jsonl"

    def spawn(*args):
        return subprocess.Popen([binary, *map(str, args)], stderr=subprocess.DEVNULL)

    aggregator = spawn(
        "aggregator", "--key", tmp_path / "aggregator.key", "--plan", plan,
        "--model", tmp_path / "initial.safetensors", "--dir", tmp_path / "aggregator",
        "--listen", addr, "--keep-spool",
    )
    helper = spawn(
        "helper", "--key", tmp_path / "helper-0.key", "--plan", plan,
        "--dir", tmp_path / "helper-0", "--connect", addr,
    )
    time.sleep(0.2)

    stakes = ft.Stakes(SOL, 10 * SOL, 3_000_000, SOL, 25, 250)
    ledger = ft.Ledger(validator, poll_ms=100)

    def site(i):
        return ft.run_site(
            key=tmp_path / f"site-{i}.key", plan=plan, grant=tmp_path / f"site-{i}.grant.json",
            connect=addr, dir=tmp_path / f"site-{i}", layout=federation.layout, ledger=ledger,
            train=lambda r, p: ({k: np.asarray(v) + 1.0 for k, v in p.items()}, 1),
        )

    with ThreadPoolExecutor(5) as pool:
        publisher = pool.submit(
            ft.run_anchor, "publisher", key=tmp_path / "sponsor.key", plan=plan,
            ledger=ft.Ledger(validator, stakes, poll_ms=100),
        )
        anchor = pool.submit(
            ft.run_anchor, "aggregator", key=tmp_path / "aggregator.key", plan=plan,
            ledger=ledger, dir=tmp_path / "aggregator",
        )
        sites = [pool.submit(site, i) for i in range(3)]
        results = [one.result(timeout=300) for one in sites]
        assert all(one["ok"] for one in results), results
        task = anchor.result(timeout=300)
        assert publisher.result(timeout=300) == task

    assert aggregator.wait(timeout=60) == 0
    assert helper.wait(timeout=60) == 0
    held = call(validator, "getAccountInfo", task, {"encoding": "base64"})["result"]["value"]
    assert held is not None, "a task account stays, as the record of the run"


def test_an_anchor_refuses_what_it_cannot_do(tmp_path):
    federation = ft.LocalFederation(
        tmp_path, {"w": np.zeros(4, dtype=np.float32)}, sites=2, did_bio="localnet"
    )
    federation.save()
    given = dict(key=tmp_path / "sponsor.key", plan=tmp_path / "plan.jsonl", ledger=ft.Ledger("http://127.0.0.1:1"))
    with pytest.raises(ValueError, match="stakes"):
        ft.run_anchor("publisher", **given)
    with pytest.raises(ValueError, match="no such role"):
        ft.run_anchor("referee", **given)
    with pytest.raises(ValueError, match="connect"):
        ft.run_anchor("watcher", **given)
