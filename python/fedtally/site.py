from . import _native
from .model import in_layout, load_model


def trainer_for(train, layout):
    """Adapts `train(round, params) -> (params, weight)` to the native call."""

    def call(round_number, base_path):
        params, weight = train(round_number, load_model(base_path))
        return in_layout(params, layout), int(weight)

    return call


def run_site(
    *,
    key,
    plan,
    grant,
    connect,
    dir,
    train,
    layout,
    revocations=None,
    give_up_secs=600,
    standing=None,
    status_dir=None,
    registry_rpc=None,
):
    """Runs one site until the run ends.

    `train(round, params)` gets the current model as read-only arrays and
    returns the tensors after local training, and a weight, usually the number
    of examples. Nothing may write to the returned arrays until the next call.

    For a plan that admits on credentials, `standing` is a JSON file with this
    site's `dataset` and `credentials`, and `status_dir` a folder holding the
    issuers' newest revocation lists. `registry_rpc` is a Solana endpoint this
    site trusts: before each round it asks whether every `did:bio` operator
    and issuer still stands, and sits the round out if one does not.
    """
    return _native.run_site_process(
        str(key),
        str(plan),
        str(grant),
        connect,
        str(dir),
        trainer_for(train, layout),
        layout,
        None if revocations is None else str(revocations),
        give_up_secs,
        None if standing is None else str(standing),
        None if status_dir is None else str(status_dir),
        registry_rpc,
    )
