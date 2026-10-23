"""Apply a scheduling-only model order while preserving the active controller.

The watchdog is briefly parked and its supervisor stopped so no in-memory
state can overwrite the edit. The watchdog then starts a fresh supervisor,
which adopts the unchanged active controller. No TPU operation is issued.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from llm_queue_state_v002 import read, save, utc

ORDER = ['qwen3_8b', 'qwen3_32b', 'gemma3_12b', 'gemma3_27b',
         'mistral_24b', 'qwen3_14b', 'mistral_7b']


def process(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=,command='],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip()


def reordered(state):
    jobs = state['jobs']
    if set(j['model'] for j in jobs) != set(ORDER):
        raise ValueError('Model set differs from the authorized seven models')
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job identity')
    before = {j['job_id']: j for j in jobs}
    result = dict(state, jobs=sorted(jobs, key=lambda j: ORDER.index(j['model'])))
    assert {j['job_id']: j for j in result['jobs']} == before
    result.update(model_order=ORDER, scheduling='model-major',
                  scheduling_updated_utc=utc(),
                  scheduling_note='Finish the active Qwen3-14B workload, then finish each model in order.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    queue = campaign / 'queue_v001'
    statepath = queue / 'state.json'
    args.output.mkdir(parents=True, exist_ok=False)
    before = read(statepath)
    proposed = reordered(before)
    summary = dict(model_order=ORDER, workloads=len(before['jobs']),
                   expected_comparisons=before['expected_comparisons'],
                   counts=dict(collections.Counter(j['status'] for j in before['jobs'])),
                   next=next(j for j in proposed['jobs'] if j['status'] == 'queued')['job_id'],
                   active_job=before.get('active_job'), applied=False)
    if not args.apply:
        save(args.output / 'summary.json', summary)
        print(json.dumps(summary))
        return
    watch = read(queue / 'watchdog-process.json')['pid']
    supervisor = read(queue / 'supervisor-process.json')['pid']
    for pid, name in [(watch, 'watch_llm_queue_v004.py'),
                      (supervisor, 'run_llm_queue_v004.py')]:
        command = process(pid)
        if name not in command or str(campaign) not in command or command.startswith('T'):
            raise RuntimeError('Expected running orchestration process not found')
    ledger = campaign / 'completed_entries.jsonl'
    digest_before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    parked = False
    try:
        os.kill(watch, signal.SIGSTOP)
        parked = True
        for _ in range(50):
            if process(watch).startswith('T'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Watchdog did not park')
        # The controller has its own session. Stopping only this supervisor
        # leaves the remote run, polling and result checkpointing untouched.
        os.kill(supervisor, signal.SIGTERM)
        for _ in range(50):
            status = process(supervisor)
            if not status or status.startswith('Z'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Supervisor did not stop; no queue edit made')
        before = read(statepath)
        if before['status'] != 'running':
            raise RuntimeError('Queue state changed; no scheduling edit made')
        active = [j for j in before['jobs'] if j['status'] == 'running']
        if len(active) != 1 or active[0]['model'] != 'qwen3_14b':
            raise RuntimeError('Expected active Qwen3-14B workload not found')
        run = campaign / active[0]['attempts'][-1]['run']
        controller = active[0]['attempts'][-1]['pid']
        command = process(controller)
        if 'run_llm_case_controller_v005.py' not in command or str(run) not in command:
            raise RuntimeError('Active controller not verified; no queue edit made')
        save(args.output / 'state-before.json', before)
        proposed = reordered(before)
        save(statepath, proposed)
        save(args.output / 'state-after.json', proposed)
        summary.update(applied=True, utc=utc(), active_controller_pid=controller,
                       watchdog_pid=watch, retired_supervisor_pid=supervisor,
                       ledger_sha256_before=digest_before,
                       ledger_sha256_after=hashlib.sha256(ledger.read_bytes()).hexdigest())
        save(args.output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    finally:
        if parked:
            os.kill(watch, signal.SIGCONT)


if __name__ == '__main__':
    main()
