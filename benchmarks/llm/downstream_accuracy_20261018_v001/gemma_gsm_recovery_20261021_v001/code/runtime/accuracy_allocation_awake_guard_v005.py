"""Prevent idle sleep on AC while the existing allocation supervisor is waiting."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


class Guard:
    def __init__(self, controller_pid, spawn=subprocess.Popen, clock=time.monotonic):
        self.controller_pid = controller_pid
        self.spawn = spawn
        self.clock = clock
        self.child = None
        self.children = []
        self.renew_at = 0

    @staticmethod
    def stop_child(child):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)

    def stop(self):
        for child in self.children:
            self.stop_child(child)
        self.children = []
        self.child = None

    def step(self, controller_matches, on_ac, allocated):
        self.children = [child for child in self.children if child.poll() is None]
        if not controller_matches or allocated:
            self.stop()
            return False
        if not on_ac:
            self.stop()
        elif (self.child is None or self.child.poll() is not None
              or self.clock() >= self.renew_at):
            # A bounded assertion expires even if this guard unexpectedly dies.
            # Popen does not acknowledge assertion registration. Let each old
            # 60-second assertion expire naturally, overlapping the new one by
            # about 20 seconds; never terminate it at renewal.
            self.child = self.spawn(
                ["/usr/bin/caffeinate", "-i", "-s", "-w",
                 str(self.controller_pid), "-t", "60"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.renew_at = self.clock() + 40
            self.children.append(self.child)
        return True


def on_ac_power(text):
    return text.splitlines()[:1] == ["Now drawing from 'AC Power'"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / args.run_id
    if run.parent != ROOT / "runs" or not run.is_dir():
        raise ValueError("Expected existing run")
    receipt = json.loads((run / "starter-start-v008.json").read_text())
    expected = " ".join(receipt["command"])
    if (str(ROOT / "runtime/start_accuracy_study_v008.py") not in receipt["command"]
            or receipt["command"][-2:] != ["--run-id", args.run_id]):
        raise ValueError("Unexpected supervisor identity")
    lock = (run / "allocation-awake-guard-v005.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard = Guard(receipt["pid"])
    previous = None

    def interrupted(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            ps = subprocess.run(
                ["ps", "-p", str(receipt["pid"]), "-o", "command="],
                capture_output=True, text=True, timeout=5,
            )
            power = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5,
            )
            matches = ps.returncode == 0 and ps.stdout.strip() == expected
            on_ac = power.returncode == 0 and on_ac_power(power.stdout)
            allocated = (run / "owned-allocation.json").exists()
            keep_running = guard.step(matches, on_ac, allocated)
            state = dict(
                utc=datetime.now(timezone.utc).isoformat(),
                controller_pid=receipt["pid"], controller_matches=matches,
                on_ac_power=on_ac, allocated=allocated,
                guard_active=guard.child is not None,
                caffeinate_pid=guard.child.pid if guard.child else None,
                active_caffeinate_pids=[child.pid for child in guard.children],
                persistent_power_settings_changed=False,
                state="watching" if keep_running else "finished",
            )
            path = run / "allocation-awake-status-v005.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(path)
            transition = (matches, on_ac, allocated, state["caffeinate_pid"])
            if transition != previous:
                print(json.dumps(state), flush=True)
                previous = transition
            if not keep_running:
                return
            time.sleep(10)
    finally:
        guard.stop()


if __name__ == "__main__":
    main()
