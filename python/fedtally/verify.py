import json

from . import _native


def verify(log, model=None) -> dict:
    """Replays a log from nothing and returns what it says.

    Raises `VerifyFailed` if any entry breaks a rule, or if `model` isn't the
    model the log ends on. `report["complete"]` is False when the log is valid
    but the run didn't finish, a site that offered a contribution was left out,
    or a round was given up for a retry or a removed helper.
    """
    text, complete = _native.verify(str(log), None if model is None else str(model))
    report = json.loads(text)
    report["complete"] = complete
    return report
