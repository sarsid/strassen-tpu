"""Read-only v010 activity derived from the large-shape mini-study evidence.

The v008 historical panels and v002 UI are retained. No remote polling, process
inspection, job control, or writes occur. A quiet journal is not proof of failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import server_v008 as original


MINI = "runs/20260921-llm-large-shapes-cohort-v001"
PHASE_NAMES = {"GRID-smoke": "Smoke checks", "GRID-screen": "Tile screening",
               "GRID-confirm": "Fresh confirmation"}
BASE_STATE = original.state


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def model_counts(models):
    """Use only counters whose evidence belongs to this specific cohort."""
    counts = []
    for model in models:
        links = model.get("evidence", [])
        belongs = any(isinstance(link, dict) and
                      str(link.get("path", "")).startswith(MINI + "/") for link in links)
        done, total = model.get("completed"), model.get("total")
        if belongs and type(done) is int and type(total) is int and 0 <= done <= total:
            counts.append(f"{model.get('name', model.get('id', 'Model'))} {done}/{total}")
    return "; ".join(counts)


def mini_activity(root, models, now):
    folder = root / MINI
    counts = model_counts(models)
    suffix = (" Completed screen/confirmation groups: " + counts + ".") if counts else ""
    result = {"kind": "activity", "id": "current", "derived_from": MINI,
              "evidence": []}
    final_path = folder / "controller-completion.json"
    final = original.read_json(final_path)
    if final:
        complete = final.get("status") == "completed" and final.get("released") is True
        result.update(title="Large-shape MM experiment completed" if complete else
                      "Large-shape MM experiment needs attention",
                      state="complete" if complete else "attention",
                      utc=final.get("finished_utc"), terminal=True,
                      detail=("All phases completed; allocation release verified." if complete else
                              "The controller recorded an incomplete or failed outcome.") + suffix,
                      next_step="Review numerical eligibility and timing evidence before drawing conclusions.",
                      evidence=[original.evidence("Controller completion", MINI + "/controller-completion.json")])
        if final.get("remote_may_still_be_running"):
            result["detail"] += " Remote work may still be running; check recovery evidence."
        elif not final.get("released"):
            result["detail"] += " Allocation release has not been verified."
        return result

    phase = next((name for name in ("GRID-confirm", "GRID-screen", "GRID-smoke")
                  if (folder / (name + "-started.json")).is_file()), None)
    if phase is None:
        return None
    started = original.read_json(folder / (phase + "-started.json"))
    result.update(title=PHASE_NAMES[phase], state="unknown", utc=started.get("started_utc"),
                  detail="Waiting for a valid execution journal." + suffix,
                  next_step="Follow the automatically refreshed execution evidence.",
                  evidence=[original.evidence("Phase start", MINI + "/" + phase + "-started.json")])
    pointer = started.get("run")
    if not isinstance(pointer, str) or not pointer.strip():
        return result
    run = Path(pointer)
    if not run.is_absolute():
        run = root / run
    run = run.resolve()
    if not run.is_relative_to((folder / "phases").resolve()) or not run.is_dir():
        result["detail"] = "Phase run path is missing or outside this cohort." + suffix
        return result
    journal = run / "artifacts/results.jsonl"
    if not journal.is_file():
        journal = run / "results.jsonl"
    data = original.READER.read(journal)
    finish_path = folder / (phase + "-finished.json")
    finish = original.read_json(finish_path)
    if not finish:
        finish_path = run / "completion.json"
        finish = original.read_json(finish_path)
    result["evidence"].append(original.evidence("Execution journal", str(journal.relative_to(root))))
    if finish:
        result["utc"] = finish.get("finished_utc") or data.get("updated_utc") or result["utc"]
        result["evidence"].append(original.evidence("Phase completion", str(finish_path.relative_to(root))))
        if finish.get("status") != "completed":
            result.update(state="attention", title=PHASE_NAMES[phase] + " needs attention",
                          detail="This phase recorded a failed or incomplete execution." + suffix)
        else:
            following = {"GRID-smoke": "Tile screening is pending.",
                         "GRID-screen": "Fresh confirmation is pending.",
                         "GRID-confirm": "Final archiving and allocation release are pending."}[phase]
            result.update(state="waiting", title=PHASE_NAMES[phase] + " completed",
                          detail=following + suffix)
        return result
    result["utc"] = data.get("updated_utc") or result["utc"]
    elapsed = original.age(result["utc"], now)
    if not data.get("updated_utc"):
        result.update(state="starting", detail="Phase launch recorded; no execution events received yet." + suffix)
    elif data.get("error"):
        result.update(state="attention", detail="The journal contains a run error; completion is pending." + suffix)
    else:
        recent = elapsed is not None and elapsed <= 120
        result.update(state="running" if recent else "stale",
                      detail=("Execution events are arriving." if recent else
                              "No recent execution event; compilation or work may still be running.") + suffix)
    if type(data.get("total")) is int:
        result["detail"] += f" This phase: {data.get('completed', 0)}/{data['total']} groups completed."
    return result


def state(root=original.ROOT, now=None):
    root = Path(root).resolve()
    now = now or datetime.now(timezone.utc)
    payload = BASE_STATE(root, now)
    activity = mini_activity(root, payload["models"], now)
    if activity is not None:
        # Once finished, later manual work updates can take over the activity card.
        manual_time = timestamp(payload["activity"].get("utc"))
        final_time = timestamp(activity.get("utc"))
        superseded = activity.get("terminal") and manual_time and final_time and manual_time > final_time
        if not superseded:
            payload["activity"] = activity
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    original.state = state
    original.UI = Path(__file__).with_name("progress_v002.html")
    print(f"Live project progress v010: http://127.0.0.1:{args.port}", flush=True)
    original.ThreadingHTTPServer(("127.0.0.1", args.port), original.Handler).serve_forever()
