"""The site side of the benchmark: Client API around the shared local update."""

import nvflare.client as flare

from common import local_update


def main():
    flare.init()
    site = int(flare.system_info()["site_name"].split("-")[1]) - 1
    while flare.is_running():
        received = flare.receive()
        updated = local_update(received.params, site)
        flare.send(
            flare.FLModel(params=updated, params_type=flare.ParamsType.FULL, current_round=received.current_round)
        )


if __name__ == "__main__":
    main()
