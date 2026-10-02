"""Local read-only status panel; artifacts, not fabricated progress, are authoritative."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]

def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result

def state():
    events = records(ROOT / "status/events.jsonl")
    agents = {}
    for entry in events:
        if entry.get("kind") == "agent":
            agents[entry["name"]] = entry
    runs = []
    for path in sorted((ROOT / "runs").glob("*/execution.json")):
        metadata = json.loads(path.read_text())
        directory = path.parent
        completion = directory / "completion.json"
        metadata["completion"] = json.loads(completion.read_text()) if completion.exists() else None
        rr = records(directory / "results.jsonl")
        if not rr:
            rr = records(directory / "artifacts/results.jsonl")
        metadata["record_count"] = len(rr)
        metadata["last_record"] = rr[-1] if rr else None
        metadata["completed_cases"] = sum(r.get("event", r.get("kind")) == "case_result" for r in rr)
        metadata["failed_cases"] = sum(r.get("event", r.get("kind")) == "case_result" and r.get("status") != "ok" for r in rr)
        runs.append(metadata)
    commits = subprocess.run(["git", "log", "-8", "--format=%h %s"], cwd=ROOT,
                             text=True, capture_output=True).stdout.splitlines()
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "agents": list(agents.values()), "events": events[-18:],
            "runs": runs, "commits": commits}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            data = json.dumps(state()).encode()
            mime = "application/json"
        elif url.path == "/":
            data = (ROOT / "status/index_v002.html").read_bytes()
            mime = "text/html; charset=utf-8"
        elif url.path.startswith("/files/"):
            target = (ROOT / unquote(url.path[7:])).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file() or any(
                part.startswith(".") for part in target.relative_to(ROOT).parts
            ):
                self.send_error(404)
                return
            data = target.read_bytes()
            mime = "text/plain; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass

if __name__ == "__main__":
    print("Status: http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
