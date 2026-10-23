"""Stage fresh Colab OAuth credentials privately; never activate or allocate.

Uses the installed client's scopes and registered installed-app localhost
redirect. The owner completes Google sign-in in the system browser. Tokens,
account details and callback codes are never printed or written to the repo.
"""
import argparse
from datetime import datetime, timezone
from importlib import resources
import json
import logging
import os
from pathlib import Path
import sys

logging.disable(logging.CRITICAL)


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    path.chmod(0o600)


def main():
    from colab_cli.auth import PUBLIC_SCOPES
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import AuthorizedSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir', type=Path, required=True)
    args = parser.parse_args()
    folder = args.private_dir.expanduser().resolve()
    store = (Path.home() / '.config/colab-cli').resolve()
    if not folder.is_relative_to(store) or folder == store:
        raise ValueError('Authentication staging must remain in the private CLI store')
    os.umask(0o077)
    folder.mkdir(mode=0o700, parents=False, exist_ok=False)
    config = json.loads(resources.files('colab_cli').joinpath('oauth_config.json').read_text())
    if config.get('installed', {}).get('redirect_uris') != ['http://localhost']:
        raise ValueError('Installed OAuth client redirect configuration differs')
    flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES)
    save(folder / 'status.json', dict(state='waiting_for_google_sign_in', started_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(state='waiting_for_google_sign_in', private_staging=True)), flush=True)
    credentials = flow.run_local_server(host='localhost', bind_addr='127.0.0.1', port=0,
        authorization_prompt_message='',
        success_message='Google sign-in received. Return to Codex to verify the account and resume the experiment.',
        open_browser=True, timeout_seconds=1200, prompt='select_account consent', access_type='offline')
    if not credentials.refresh_token:
        raise ValueError('Google did not provide a refresh credential')
    with AuthorizedSession(credentials, refresh_timeout=20) as session:
        response = session.get('https://www.googleapis.com/oauth2/v2/userinfo', timeout=30)
        response.raise_for_status()
        user = response.json()
    if not user.get('id') or not user.get('email') or user.get('verified_email') is not True:
        raise ValueError('Could not verify the signed-in Google account')
    save(folder / 'token.json', json.loads(credentials.to_json()))
    save(folder / 'account.json', dict(subject=user['id'], email=user['email'], verified=True))
    save(folder / 'completed.json', dict(state='authenticated_pending_activation',
        finished_utc=datetime.now(timezone.utc).isoformat(), credentials_activated=False))
    print(json.dumps(dict(state='authenticated_pending_activation', account_verified=True,
                          credentials_activated=False, allocated_runtime=False)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # OAuth errors can contain callback URLs/codes. Emit only their type.
        print(json.dumps(dict(state='authentication_failed', error_type=type(error).__name__)), flush=True)
        sys.exit(1)
