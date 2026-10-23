"""Release exactly the completed campaign's allocation after local archival."""
import argparse
import json
from pathlib import Path
import colab_control_v001 as control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--expect-endpoint', required=True)
    parser.add_argument('--config', type=Path, default=control.DEFAULT_CONFIG)
    parser.add_argument('--token-file', type=Path, default=control.DEFAULT_TOKEN)
    args = parser.parse_args()
    with control.api(args.token_file) as client:
        store, saved = control.refreshed_session(args, client)
        client.unassign(saved.endpoint)
        if any(item.endpoint == saved.endpoint for item in client.list_assignments()):
            raise RuntimeError('Unassignment requested but endpoint is still listed; preserve session record')
        store.remove(saved.name)
        # The existing bounded heartbeat exits when its saved session is absent.
        control.emit({'kind': 'allocation_released', 'session': saved.name,
                      'endpoint': saved.endpoint, 'verified_absent': True})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        control.emit({'kind': 'release_error', **control.safe_error(error)})
        raise SystemExit(1)
