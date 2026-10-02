# Colab v5e runtime control: inspected interfaces

Investigation date: 2026-09-19. This document records existing local interfaces; no TPU was allocated, restarted, or used by this investigation.

## Current state

A read-only authenticated backend query returned **no active assignments**. The local session map was empty. Existing OAuth credentials worked. The sandboxed query could not reach the service; the same read-only query with network escalation succeeded.

The installed client is at `../../.venv-colab/lib/python3.12/site-packages/colab_cli/` relative to this directory. Invoke it from the parent workspace as `.venv-colab/bin/colab`. No browser interaction is necessary while the existing credentials remain valid.

## Lifecycle and reuse

From the parent workspace, with a campaign-specific session name:

```sh
.venv-colab/bin/colab --auth oauth2 sessions
.venv-colab/bin/colab --auth oauth2 new -s strassen-mm-focus-v5e-20260919 --tpu v5e1
.venv-colab/bin/colab --auth oauth2 status -s strassen-mm-focus-v5e-20260919
```

Only call `new` if the desired live allocation does not already exist. `sessions` synchronizes local metadata and can prune stale local entries; use the read-only API below when a nonmutating check is required. Always name the session explicitly. Avoid `colab run`: it provisions a fresh runtime for each job and usually releases it afterward.

`new` automatically starts a detached keep-alive process. The installed implementation pings once per 60 seconds, stops after 24 hours, and stops after two consecutive HTTP 4xx failures. Its PID is stored in the local session state. This does not guarantee Colab will preserve a runtime indefinitely.

Once all authorized work and artifact retrieval are complete, the lifecycle owner can release the campaign allocation with:

```sh
.venv-colab/bin/colab --auth oauth2 stop -s strassen-mm-focus-v5e-20260919
```

Do not stop/recreate a runtime between N1–N4 stages merely to reset Python state. Use fresh sequential child processes on the same VM. Do not run TPU benchmark agents simultaneously on the device.

## Upload, asynchronous execution and retrieval

Use unique, immutable run directories and archive filenames. The CLI file API overwrites a destination if it already exists, so check local/remote destination existence before transfer.

```sh
.venv-colab/bin/colab --auth oauth2 upload -s SESSION Strassen_MM_Focus/runs/RUN_ID/source.tar.gz /content/Strassen_MM_Focus/RUN_ID/source.tar.gz
.venv-colab/bin/colab --auth oauth2 exec -s SESSION -f Strassen_MM_Focus/runtime/launch_RUN_ID.py --timeout 60
.venv-colab/bin/colab --auth oauth2 download -s SESSION /content/Strassen_MM_Focus/RUN_ID/artifacts.tar.gz Strassen_MM_Focus/runs/RUN_ID/artifacts.tar.gz
```

Create the remote parent directory before upload. `exec -f` sends the local script contents into the existing Jupyter kernel; it does not upload the script as a file. The installed CLI supports `--timeout`, default 30 seconds. For long jobs, use a small launcher that starts a detached remote subprocess and immediately returns its PID:

```python
from pathlib import Path
import subprocess, sys
run = Path('/content/Strassen_MM_Focus/RUN_ID')
# Source extraction and verification occur before launching.
# The worker records progress and a final success/failure manifest.
with (run / 'launcher.log').open('x') as log:
    child = subprocess.Popen(
        [sys.executable, '-u', str(run / 'source' / 'worker.py')],
        cwd=run / 'source', stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
print({'pid': child.pid, 'run_id': run.name})
```

Poll small progress records through the Jupyter Contents API, and retrieve final archives before starting the next phase. Archive failed attempts as well as successes. Source snapshots should include their SHA256 manifest, source commit, experiment plan, environment, and runtime identity; results should include raw timing samples, execution order, all errors, and terminal exit status. Run directories must be created with `exist_ok=False`; use new IDs for reruns. A status dashboard can be regenerated, but experiment evidence must remain immutable.

Do not initialize JAX in the long-lived Jupyter parent kernel. A child process should own the TPU, exit, and release it before the next stage. Historical launchers already follow this approach.

## Direct Python API for safe status and token refresh

Public interfaces in the installed local code:

- `colab_cli.client.Client(Prod(), authorized_session)`
- `client.list_assignments()` returns models with `endpoint`, `accelerator`, `variant`, `machine_shape`, and `runtime_proxy_info`.
- `client.assign(uuid.UUID(...), variant=Variant.TPU, accelerator=Accelerator.V5E1)` allocates; lifecycle owner only.
- `client.keep_alive_assignment(endpoint)` refreshes activity; the installed method intentionally treats a read timeout as a successful ping.
- `client.unassign(endpoint)` releases; lifecycle owner only.
- `colab_cli.runtime.ColabRuntime(url, token, kernel_id=..., session_id=...)` followed by `execute_code(code, timeout=..., output_hook=...)` connects to an existing runtime.
- `colab_cli.contents.ContentsClient(session_state).list_dir(path)`, `.upload(local, remote)`, `.download(remote, local)` use the Jupyter Contents API. This client has no explicit request timeout in the installed implementation.

A read-only allocation query can refresh credentials in memory without invoking the CLI login flow or changing the saved session state:

```python
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession
from colab_cli.client import Client, Prod
creds = Credentials.from_authorized_user_file(
    str(Path.home() / '.config/colab-cli/token.json'))
with AuthorizedSession(creds, refresh_timeout=15) as session:
    assignments = Client(Prod(), session).list_assignments()
    # Print only this sanitized projection, never the assignment model.
    print([{'endpoint': a.endpoint, 'accelerator': a.accelerator.value}
           for a in assignments])
```

The installed CLI `exec` and `ContentsClient` use the saved runtime proxy token. For a long campaign, get a fresh `runtime_proxy_info` via `list_assignments()` if the token expires, match **the same endpoint**, and use the new URL/token for the connection or update the local session state without changing its endpoint/kernel identifiers. A new proxy token is not evidence of a new VM. Do not interpret a stale-token 401 alone as proof the allocation vanished: check the assignment API first.

Credentials and raw session metadata remain in `~/.config/colab-cli/`; never copy those files into the repository or dashboard. Avoid `--logtostderr`. The installed CLI's debug log contains HTTP headers and responses and is not suitable for publication or artifact collection. If catching requests/auth exceptions, print only the exception class and sanitized HTTP status, not raw URLs, headers, bodies, or traceback objects that might contain tokens.

## Consistent machine and software

Keep one live v5e allocation for the whole authorized N1–N4 campaign if possible. At campaign start and every run, capture and compare:

- Sanitized Colab endpoint and accelerator type.
- Remote `platform.node()` hostname and `/proc/sys/kernel/random/boot_id` when available.
- Python executable/version, platform, package versions and environment flags.
- JAX backend, device count, device kind, local/global IDs and available coordinates/slice index.
- Source/plan SHA256 values and the run sequence.

Historical compatible stack: JAX **0.7.2**, jaxlib **0.7.2**, libtpu **0.0.21.1**. One recorded runtime used Python 3.13.15, NumPy 2.1.3 and ml_dtypes 0.6.0. Treat these as inspected historical versions, not evidence that an unprovisioned new runtime already has them. Install and verify the chosen stack once, then freeze it across the campaign. The recorded v5e JAX device string is `TPU v5 lite` with one device.

Record baseline anchors at the beginning and end of every phase, randomize/interleave candidate order, exclude compilation/warmup from steady-state timing, and block until TPU completion for each measured call. Same hardware does not eliminate drift from compiler caches, host activity, or device/runtime state.

Colab does not expose a supported reservation of the same physical TPU across termination/reallocation through the inspected client. Endpoint/hostname/boot ID continuity provide **runtime continuity**, not an independently verified permanent silicon serial number. A browser reconnect may retain the same runtime; a new allocation cannot be assumed to do so. If these identity fields change, start a new machine cohort, rerun anchors and comparisons needed for that cohort, and never silently pool it with the original machine's results.

## Inspected provenance

- `.venv-colab/lib/python3.12/site-packages/colab_cli/{client,runtime,contents,state,common,auth,cli}.py`
- `.venv-colab/lib/python3.12/site-packages/colab_cli/commands/{session,execution,files}.py`
- `colab-setup/tpu_usage_monitor.py`: sanitized read-only status implementation.
- `colab-setup/runtime_probe.py`: probe in a short-lived child.
- `colab-setup/finalization-v5e/run_confirm_1.py`: archive verification, sequential child execution and failure-preserving artifacts.
- `colab-setup/equal-tuning-v5e/equal-tuning-screen/runtime.json`: historical version provenance.
