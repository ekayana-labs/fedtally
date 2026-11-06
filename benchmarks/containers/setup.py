"""One run's keys, plan and grants, written under /run for its parties."""

import json
import struct
import subprocess
import sys
from array import array

SITES, HELPERS, ROUNDS, COORDS, MIN_COHORT = map(int, sys.argv[1:])
PURPOSE = "containers"


def fedtally(*args):
    done = subprocess.run(["fedtally", *args], check=True, capture_output=True, text=True)
    return done.stdout


def key(name, *extra):
    return json.loads(fedtally("keygen", "--out", f"/run/{name}.key", *extra))


sponsor = key("sponsor")
aggregator = key("aggregator", "--tls")
helpers = [key(f"helper-{h}") for h in range(HELPERS)]
sites = [key(f"site-{s}") for s in range(SITES)]

# The form fedtally writes itself, which it also checks a model file against.
header = json.dumps(
    {"w": {"data_offsets": [0, COORDS * 4], "dtype": "F32", "shape": [COORDS]}},
    separators=(",", ":"),
).encode()
header += b" " * (-len(header) % 8)
with open("/run/init.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(header)))
    f.write(header)
    array("f", [0.0] * COORDS).tofile(f)

spec = {
    "purpose": PURPOSE,
    "aggregator": {"did": aggregator["did"], "tls_pin": aggregator["tls_pin"]},
    "helpers": [
        {"did": h["did"], "seal": h["seal"], "seal_proof": h["seal_proof"]} for h in helpers
    ],
    "sites": [s["did"] for s in sites],
    "layout": {"tensors": [{"name": "w", "dtype": "F32", "shape": [COORDS]}]},
    "rounds": ROUNDS,
    "min_cohort": MIN_COHORT,
    "chunk_coords": 4096,
    "encoding": {"frac_bits": 24, "clip": 128.0},
    "max_weight": 1000,
    "timeouts": {"join_secs": 120, "collect_secs": 60, "attest_secs": 120, "ack_secs": 120},
}
with open("/run/spec.json", "w") as f:
    json.dump(spec, f)
fedtally(
    "plan", "--key", "/run/sponsor.key", "--spec", "/run/spec.json",
    "--model", "/run/init.safetensors", "--out", "/run/plan.jsonl",
)
for s, site in enumerate(sites):
    fedtally(
        "grant", "--key", f"/run/site-{s}.key", "--to", sponsor["did"],
        "--purpose", PURPOSE, "--any-helper", "--min-cohort", str(MIN_COHORT),
        "--max-rounds", str(ROUNDS), "--out", f"/run/site-{s}.ucan",
    )
    fedtally(
        "invoke", "--key", "/run/sponsor.key", "--plan", "/run/plan.jsonl",
        "--site", site["did"], "--grant", f"/run/site-{s}.ucan",
        "--out", f"/run/site-{s}.grant.json",
    )
