"""A whole federation on one machine, over real TLS on loopback."""

from dataclasses import dataclass
from pathlib import Path

from . import _native
from .model import layout_of, load_model, save_model
from .site import trainer_for
from .verify import verify


def default_min_cohort(sites: int) -> int:
    """Three quarters of the sites, since the aggregator chooses who is in a round."""
    return max(min(sites, 2), -(-sites * 3 // 4))


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
    `train(site, round, params)` returns `(params, weight)`.
    """

    def __init__(
        self,
        root,
        initial: dict,
        *,
        sites: int,
        helpers: int = 1,
        rounds: int = 1,
        min_cohort: int | None = None,
        chunk_coords: int = 1 << 19,
        frac_bits: int = 24,
        clip: float = 128.0,
        max_weight: int = 1 << 20,
        audits: int = 1,
        purpose: str = "local run",
        granted_purpose: str | None = None,
        join_secs: int = 120,
        collect_secs: int = 600,
        attest_secs: int = 300,
        ack_secs: int = 120,
        sig_alg: str = "ed25519",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.layout = layout_of(initial)
        first = self.root / "initial.safetensors"
        save_model(first, initial)
        self.native = _native.Federation(
            str(self.root),
            self.layout,
            str(first),
            sites,
            helpers,
            rounds,
            min_cohort if min_cohort is not None else default_min_cohort(sites),
            chunk_coords,
            frac_bits,
            clip,
            max_weight,
            audits,
            purpose,
            granted_purpose,
            join_secs,
            collect_secs,
            attest_secs,
            ack_secs,
            sig_alg,
        )
        self.sites = sites

    @property
    def log_path(self) -> str:
        return self.native.log_path

    def save(self) -> None:
        self.native.save()

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
