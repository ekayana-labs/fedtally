# fedtally

Federated learning for biomedical data that every party can check.

Sites train on their own data. A run admits them on credentials, their updates are masked and carry noise that bounds what the model can reveal about any one site, and an aggregator adds up vectors it can't read. Every party signs a small log. Each site checks the result before accepting it, and anyone holding the log can check it later without contacting or trusting the aggregator. The aggregator can be killed at any point and the run carries on when it restarts.

This is an implementation of the EkayanaFL protocol, changed where another mechanism gives the same property at lower cost. The core is Rust. Trainers are ordinary Python.

**Status: early and unreviewed.** It has had no security audit and has never run on real data. Independent reads of the design have found serious flaws that were then fixed, which is a good reason to expect more. The protocol will change.

## Try it

You need Rust and Python 3.11 or later, on macOS or Linux.

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install maturin numpy && maturin develop --release
```

```python
import numpy as np
import fedtally as ft

start = {"w": np.zeros(12, np.float32), "b": np.zeros(1, np.float32)}

def train(site, round_number, params):
    updated = {name: np.asarray(value) + 0.1 for name, value in params.items()}
    return updated, 100  # new parameters, and a weight such as the number of examples

run = ft.LocalFederation("run", start, sites=4, helpers=1, rounds=5).run(train)
print(run.model["w"][:3])
print(ft.verify(run.log_path, run.model_path)["complete"])
```

This runs everything on one machine over real TLS. The same class takes the rest of the protocol as options:

```python
private = ft.LocalFederation(
    "private", start, sites=8, rounds=5,
    privacy=ft.Privacy(epsilon=3.0, delta=1e-5, clip=1.0),  # noise at every site, a budget in the log
    credentials=True,   # admit sites on an affiliation, an ethics approval and an access grant
    sample=6,           # draw six sites a round from a beacon anyone can recompute
    bounded=True,       # each site proves its update is short, or is left out of the round
    did_bio="devnet",   # name every party by did:bio
).run(train)
print(private.report["privacy"]["epsilon_spent"])
```

For separate processes, see `fedtally --help`, `ft.run_site`, and `crates/fedtally-cli/tests/cli.rs`, which goes through the setup one party at a time. [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) trains a small model, looks at what the aggregator holds, tampers with the log, runs with privacy and revokes a credential.

To run the tests:

```bash
cargo test
```

```bash
pip install -e ".[dev]" && cargo build --release -p fedtally-cli && FEDTALLY_BIN=target/release/fedtally pytest
```

The ledger program has its own workspace and needs the Solana tools. See [ledger/README.md](ledger/README.md).

## What it does

| | How | Where |
|---|---|---|
| **Who may take part** | A rule over W3C Verifiable Credentials, for sites, their datasets and the operators. GA4GH visas are carried as issued. Dataset terms in GA4GH DUO codes must allow the run's stated purpose | `credential.rs`, `policy.rs`, `duo.rs` |
| **Taking permission back** | Issuers sign W3C Bitstring Status Lists. A revoked site is out from the next round, and the log shows when and why | `status.rs`, `replay.rs` |
| **What a site agrees to** | The site's own key signs its conditions as a UCAN grant: purpose, helpers, cohort size, privacy budget. Its machine refuses any plan outside them | `admission.rs` |
| **Who can read an update** | Nobody, unless the aggregator colludes with every helper | `vector.rs`, `model.rs` |
| **What the model reveals** | Each site clips its update and adds discrete Gaussian noise. The budget is one integer in the log, and a round that would overspend it doesn't open | `privacy.rs` |
| **What a dishonest site can send** | Each site proves a bound on the length of its masked update to the aggregator and helpers together. One that can't is left out of the round | `norm.rs` |
| **Whether the sum is right** | Linear fingerprints over vectors committed before the challenge exists, checked and signed by every site in the round | `replay.rs` |
| **Who is in a round** | Fixed by the plan, or drawn each round from a beacon over the model | `sampling.rs` |
| **Identities** | `did:key`, or `did:bio` with a check against the Solana registry before each round | `keys.rs`, `registry.rs` |
| **Who is paid** | Equal shares, or shares by score for groups of sites, with the privacy cost of scoring charged to the budget | `scoring.rs` |
| **A neutral arbiter** | A Solana program holds stake and rewards, mirrors the budget and the draw, and slashes the aggregator when one leaf of a Merkle tree shows the aggregate is not the sum. `fedtally anchor` posts a run to it, and a watcher disputes a wrong round | `ledger/`, `crates/fedtally-chain` |
| **Record of a run** | Signed, hash chained and replayable offline | `log.rs`, `report.rs` |

[docs/design.md](docs/design.md) has the protocol, why each check is sound, and a table of what each adversary can and can't do.

## How the design differs from NVFlare and Flower

| | NVFlare | Flower | fedtally |
|---|---|---|---|
| **Who can read an update** | The server, by default. Homomorphic encryption is an option. No secure aggregation found | The server, by default. SecAgg+ needs the legacy API, and seeds its masks from 32 bits (`secaggplus_utils.py:87-92`) | Nobody, unless the aggregator colludes with every helper |
| **Can a site check the aggregate** | No mechanism found | No mechanism found | Yes |
| **Differential privacy** | Filters a site applies to its own output | Clipping and noise added by the server, or by each client alone | Noise added by every site under masking, so no server is trusted to add it, with one budget every party enforces |
| **Server killed mid-round** (*measured*) | The job doesn't resume, and stays `RUNNING` | Default flags lose the run. Process isolation with `--database` recovers if the ServerApp survives and the outage stays under about a minute | Recovered in 8 of 8 trials, outages up to 300 s, with no site training twice |
| **Server memory at 1 GiB, 4 sites** (*measured*) | 13.5 GiB | 13.0 GiB | 0.23 GiB, plus 17 GiB of disk |
| **Code on the server** | Jobs can bring code, by policy | Apps are installed and run | None. Fixed Rust binaries |
| **Who decides a site takes part** | A provisioning CA and role policies | A server-side key list, with accounts in the paid product | Credentials from named issuers, and the site's own signed conditions |
| **Record of a run** | Plain text log | Paid feature | Signed, hash chained and replayable offline |
| **Post-quantum** | None found | None found | ML-DSA signatures for records and credentials as an option, X25519 with ML-KEM for seeds and TLS |

The difference underneath is that **the log is the only control state**. Aggregator, helpers, sites and auditors all fold the same signed records through the same pure function and react to the result, so a restart is a replay and an auditor is just a site with no data.

## What it doesn't do yet

- Anything but FedAvg over float32 and float64 tensors. No server optimisers, asynchronous rounds or personalisation.
- An exact norm bound for large models. Above roughly a hundred thousand parameters the proof holds a dishonest site to a multiple of the bound, and a site working with a helper can pass anything.
- Scores for single sites, which would contradict the privacy guarantee. Scored groups can't be anchored on the ledger yet, which pays equal shares.
- Run on a public cluster. The ledger program and its clients are tested on a local validator only.
- Dropouts after a round is closed. A site that drops then must return within the deadline or the run stops. Any single site or helper can stop a run this way.
- Failover. The aggregator is one process, and restarting it is the only recovery. A helper needs durable disk.
- Cross device use. It's built for tens of sites.
- Fed-Camelyon16 from the paper's evaluation. The tabular FLamby tasks and the attack suite are in `benchmarks/`.
- Windows, GPU aware transfer, compression, dashboards, or Kubernetes.

## Layout

| Path | What |
|---|---|
| `crates/fedtally-core` | Field, fingerprints, privacy, credentials, signed records and the replay rules |
| `crates/fedtally-node` | Aggregator, helper and site |
| `crates/fedtally-chain` | Posts a run to the ledger program, and the watcher |
| `crates/fedtally-cli` | The `fedtally` command |
| `crates/fedtally-py`, `python/` | Python bindings and SDK |
| `ledger/` | The Solana program, in its own workspace |
| `benchmarks/` | Harnesses for all three systems, the privacy evaluation, and raw results |

## Licence

Apache-2.0.