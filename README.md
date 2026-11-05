# fedtally

Federated averaging that every party can check.

Sites train on their own data. An aggregator adds up vectors it can't read, and every party signs a small log. Each site checks the result before accepting it, and anyone holding the log can check it later without contacting or trusting the aggregator. The aggregator can be killed at any point and the run carries on when it restarts.

The core is Rust. Trainers are ordinary Python.

**Status: early and unreviewed.** It has had no security audit and has never run on real data. An independent read of the design found serious flaws that have since been fixed, which is a good reason to expect more. The protocol will change.

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

This runs everything on one machine over real TLS. For separate processes, see `fedtally --help`, `ft.run_site`, and `crates/fedtally-cli/tests/cli.rs`, which goes through the setup one party at a time. [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) trains a small model, looks at what the aggregator holds, and tampers with the log.

To run the tests:

```bash
cargo test
```

```bash
pip install -e ".[dev]" && cargo build --release -p fedtally-cli && FEDTALLY_BIN=target/release/fedtally pytest
```

## How the design differs from NVFlare and Flower

| | NVFlare | Flower | fedtally |
|---|---|---|---|
| **Who can read an update** | The server, by default. Homomorphic encryption is an option. No secure aggregation found | The server, by default. SecAgg+ needs the legacy API, and seeds its masks from 32 bits (`secaggplus_utils.py:87-92`) | Nobody, unless the aggregator colludes with every helper. Masks are ChaCha12 with 256 bit seeds |
| **Can a site check the aggregate** | No mechanism found | No mechanism found | Yes. Linear fingerprints over vectors committed before the challenge exists |
| **Server killed mid-round** (*measured*) | The job doesn't resume, and stays `RUNNING` | Default flags lose the run. Process isolation with `--database` recovers if the ServerApp survives and the outage stays under about a minute | Recovered in 8 of 8 trials, outages up to 300 s, with no site training twice |
| **Server memory at 1 GiB, 4 sites** (*measured*) | 13.5 GiB | 13.0 GiB | 0.18 GiB, plus 17 GiB of disk |
| **Code on the server** | Jobs can bring code, by policy | Apps are installed and run | None. Fixed Rust binaries |
| **Who decides a site takes part** | A provisioning CA and role policies | A server-side key list, with accounts in the paid product | The site's own key. It delegates to a sponsor with conditions, using [pq-ucan](https://github.com/ekayana-labs/pq-ucan), and can revoke between rounds |
| **Record of a run** | Plain text log | Paid feature | Signed, hash chained and replayable offline |
| **Post-quantum** | None found | None found | ML-DSA signatures as an option, X25519 with ML-KEM for seeds and TLS |

The difference underneath is that **the log is the only control state**. Aggregator, helpers, sites and auditors all fold the same signed records through the same pure function and react to the result, so a restart is a replay and an auditor is just a site with no data. [docs/design.md](docs/design.md) has the protocol, why the check is sound, and a table of what each adversary can and can't do.

## What it doesn't do yet

- Anything but weighted FedAvg over float32 and float64 tensors. No server optimisers, asynchronous rounds or personalisation.
- Differential privacy or robust aggregation. A masked update can't be range checked, so a site can send anything, and one bad site can poison the model.
- Dropouts after a round is closed. A site that drops then must return within the deadline or the run stops, and you start a new plan from the last sealed model. Any single site or helper can stop a run this way.
- Failover. The aggregator is one process, and restarting it is the only recovery. A helper needs durable disk, for the reason in the design doc.
- Cross device use. It's built for tens of sites and tested to 16.
- Windows, GPU aware transfer, compression, dashboards, or Kubernetes.

## Layout

| Path | What |
|---|---|
| `crates/fedtally-core` | Field, fingerprints, signed records and the replay rules |
| `crates/fedtally-node` | Aggregator, helper and site |
| `crates/fedtally-cli` | The `fedtally` command |
| `crates/fedtally-py`, `python/` | Python bindings and SDK |
| `benchmarks/` | Harnesses for all three systems, and raw results |

## Licence

Apache-2.0. No code was taken from NVFlare or Flower. The benchmark harnesses use them as ordinary dependencies.
