"""A whole federation on one machine, over real TLS on loopback."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import _native
from .model import layout_of, load_model, save_model
from .site import trainer_for
from .verify import verify


@dataclass
class Privacy:
    """A privacy target for a whole site's data.

    Every site clips its update to an L2 norm of `clip`, in the model's own
    units, and adds noise before masking. The run stops rather than spend more
    than `epsilon` at `delta`. `colluding` is how many sites in a round are
    assumed to give their noise away.
    """

    epsilon: float
    delta: float
    clip: float
    colluding: int = 0


@dataclass
class RunResult:
    model: dict | None
    model_path: str | None
    log_path: str
    report: dict
    sites: list
    aggregator_error: str | None

    @property
    def ok(self) -> bool:
        return self.aggregator_error is None and all(s["ok"] for s in self.sites)


class LocalFederation:
    """Keys, plan and grants for every party, in one directory.

    A real deployment makes these separately and no party holds another's key.
    `train(site, round, params)` returns `(params, weight)`. With `privacy`
    set, every site counts the same and the weight is ignored.
    """

    def __init__(
        self,
        root,
        initial: dict,
        *,
        sites: int,
        helpers: int = 1,
        rounds: int = 1,
        privacy: Privacy | None = None,
        **options,
    ):
        """`options` are passed through: `min_cohort`, `chunk_coords`,
        `frac_bits`, `clip`, `max_weight`, `audits`, `purpose`,
        `granted_purpose`, `sig_alg`, the four `*_secs` timeouts, `sample`
        (how many sites each round draws), `credentials` (admit sites only
        on credentials from a university, a review board and a data access
        committee, which this class then plays), `did_bio` (name every
        party by `did:bio` on that network), `bounded` (every site proves its
        update is within the privacy terms, and one that can't is left out),
        `scoring` (split each cohort into groups of at least this many sites
        and score them for rewards) and `cheats` (a map from a site's number
        to how many times past the clip it reaches, for attack tests)."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.layout = layout_of(initial)
        first = self.root / "initial.safetensors"
        save_model(first, initial)
        given = {k: v for k, v in options.items() if v is not None}
        given.update(sites=sites, helpers=helpers, rounds=rounds)
        if privacy is not None:
            given["privacy"] = asdict(privacy)
        self.native = _native.Federation(
            str(self.root), self.layout, str(first), json.dumps(given)
        )
        self.sites = sites

    @property
    def log_path(self) -> str:
        return self.native.log_path

    def save(self) -> None:
        self.native.save()

    def revoke_access(self, site: int) -> None:
        """Takes back a site's access grant. It is out from the next round."""
        self.native.revoke_access(site)

    def run(self, train, keep_spool: bool = False) -> RunResult:
        def for_site(i):
            return trainer_for(lambda r, params: train(i, r, params), self.layout)

        out = self.native.run([for_site(i) for i in range(self.sites)], keep_spool)
        finished = [s for s in out["sites"] if s["ok"]]
        path = finished[0]["final_model"] if finished else None
        return RunResult(
            model=load_model(path) if path else None,
            model_path=path,
            log_path=self.log_path,
            report=verify(self.log_path) if Path(self.log_path).exists() else {},
            sites=out["sites"],
            aggregator_error=out["aggregator_error"],
        )
