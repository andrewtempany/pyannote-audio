"""Local front end for launching harness runs and watching their progress.

Zero extra dependencies -- uses only the Python standard library
(http.server) so it runs with the same interpreter as the rest of the
harness. Not for production/multi-user use: single local user, one run at a
time, no auth. Wraps run_experiment.sh (which itself wraps run_harness.py)
so this UI can never drift from the tracked-run workflow that script
already encodes -- there is exactly one place that knows how to invoke the
harness correctly.

Progress tracking (deliberately "option 1", coarse-but-easy): the harness
loop itself isn't touched. Instead this launcher passes a fresh --cache-dir
per run (a subdirectory under .harness_runs_ui/<run_id>/cache) so counting
*.rttm files in that directory is an exact, unambiguous "N of TOTAL files
done" signal -- no need to parse cache-key hashes or guess which cached
file belongs to which run.

Usage:
    python harness_ui/server.py
    (then open http://localhost:8765 in a browser)
"""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_UI_DIR = REPO_ROOT / ".harness_runs_ui"
RUNS_UI_DIR.mkdir(exist_ok=True)

# in-memory registry of runs this server process has launched
_runs: dict[str, dict] = {}
_lock = threading.Lock()


def _total_file_count(data_root: str, condition: str, split: str) -> int:
    """Count expected files from the split's .rttm (one line per segment,
    but uris repeat) -- count *distinct* uris instead."""
    rttm_path = Path(data_root) / condition / f"{split}.rttm"
    if not rttm_path.exists():
        return 0
    uris = set()
    with open(rttm_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                uris.add(parts[1])
    return len(uris)


def _launch_run(run_id: str, run_condition: str, notes: str, extra_args: list[str],
                 data_root: str, condition: str, split: str, oracle_rttm: str | None = None) -> None:
    cache_dir = RUNS_UI_DIR / run_id / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RUNS_UI_DIR / run_id
    log_path = out_dir / "log.txt"

    total = _total_file_count(data_root, condition, split)
    with _lock:
        _runs[run_id]["total"] = total
        _runs[run_id]["status"] = "running"
        _runs[run_id]["started_at"] = time.time()

    cmd = [
        "python", "run_harness.py",
        "--data-root", data_root,
        "--condition", condition,
        "--split", split,
        "--device", "cuda",
        "--cache-dir", str(cache_dir),
        "--dotenv", ".env",
        "--per-file-csv", str(out_dir / "per_file.csv"),
        "--summary", str(out_dir / "summary.json"),
        "--run-condition", run_condition,
        "--runs-dir", "runs",
        "--notes", notes,
        *extra_args,
    ]
    if oracle_rttm:
        cmd += ["--oracle-rttm", oracle_rttm]

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        )
        with _lock:
            _runs[run_id]["pid"] = proc.pid
            _runs[run_id]["_proc"] = proc
        proc.wait()

    with _lock:
        if _runs[run_id]["status"] == "killing":
            _runs[run_id]["status"] = "killed"
        else:
            _runs[run_id]["status"] = "done" if proc.returncode == 0 else "failed"
        _runs[run_id]["returncode"] = proc.returncode
        _runs[run_id]["finished_at"] = time.time()

    if proc.returncode == 0:
        try:
            from harness.aggregate_runs import aggregate_runs
            aggregate_runs(runs_dir="runs", output_csv_path="runs/comparison.csv")
        except Exception as exc:  # noqa: BLE001 -- surface any failure into the run record
            with _lock:
                _runs[run_id]["aggregate_error"] = str(exc)


def _progress(run_id: str) -> dict:
    with _lock:
        run = dict(_runs.get(run_id, {}))
    run.pop("_proc", None)  # not JSON-serializable, and internal-only
    if not run:
        return {"error": "unknown run_id"}

    cache_dir = RUNS_UI_DIR / run_id / "cache"
    done = len(list(cache_dir.glob("*.rttm"))) if cache_dir.exists() else 0
    run["done"] = done
    return run


def _kill_run(run_id: str) -> dict:
    with _lock:
        run = _runs.get(run_id)
        if not run:
            return {"error": "unknown run_id"}
        proc = run.get("_proc")
        if not proc or run.get("status") not in ("starting", "running"):
            return {"error": "run is not currently active"}
        run["status"] = "killing"

    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()  # didn't die gracefully -- force it
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the server
        return {"error": f"failed to kill: {exc}"}

    return {"status": "killing"}


def _read_comparison_csv() -> dict:
    csv_path = REPO_ROOT / "runs" / "comparison.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {"columns": [], "rows": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    return {"columns": columns, "rows": rows}


def _gpu_status() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except FileNotFoundError:
        return {"error": "nvidia-smi not found -- no NVIDIA GPU/driver detected"}
    except subprocess.CalledProcessError as exc:
        return {"error": f"nvidia-smi failed: {exc.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}

    gpus = []
    for line in result.stdout.strip().splitlines():
        name, util, mem_used, mem_total, temp = [p.strip() for p in line.split(",")]
        gpus.append({
            "name": name,
            "utilization_pct": float(util),
            "memory_used_mib": float(mem_used),
            "memory_total_mib": float(mem_total),
            "temperature_c": float(temp),
        })
    return {"gpus": gpus, "checked_at": time.time()}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        html_path = Path(__file__).parent / "index.html"
        body = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- stdlib method name
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_html()
            return

        if parsed.path == "/api/runs":
            with _lock:
                run_ids = list(_runs.keys())
            self._send_json({"runs": [_progress(rid) for rid in run_ids]})
            return

        if parsed.path == "/api/progress":
            qs = parse_qs(parsed.query)
            run_id = qs.get("run_id", [None])[0]
            if not run_id:
                self._send_json({"error": "run_id required"}, status=400)
                return
            self._send_json(_progress(run_id))
            return

        if parsed.path == "/api/comparison":
            self._send_json(_read_comparison_csv())
            return

        if parsed.path == "/api/gpu":
            self._send_json(_gpu_status())
            return

        if parsed.path == "/api/log":
            qs = parse_qs(parsed.query)
            run_id = qs.get("run_id", [None])[0]
            log_path = RUNS_UI_DIR / (run_id or "") / "log.txt"
            if not run_id or not log_path.exists():
                self._send_json({"error": "no log for that run_id"}, status=404)
                return
            tail_lines = log_path.read_text(errors="replace").splitlines()[-200:]
            self._send_json({"log": "\n".join(tail_lines)})
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 -- stdlib method name
        parsed = urlparse(self.path)

        if parsed.path == "/api/kill":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON body"}, status=400)
                return
            run_id = payload.get("run_id")
            if not run_id:
                self._send_json({"error": "run_id required"}, status=400)
                return
            result = _kill_run(run_id)
            self._send_json(result, status=400 if "error" in result else 200)
            return

        if parsed.path != "/api/launch":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        run_condition = payload.get("run_condition", "baseline")
        notes = payload.get("notes", "").strip()
        refinement_strategy = payload.get("refinement_strategy", "identity")
        oracle_scope = payload.get("oracle_scope")
        counts_toward_results = bool(payload.get("counts_toward_results", False))
        data_root = payload.get("data_root") or str(
            Path.home() / "Code" / "AMI-diarization-setup" / "harness-data"
        )
        condition = payload.get("condition", "IHM")
        split = payload.get("split", "test")
        oracle_rttm = payload.get("oracle_rttm")

        if not notes:
            self._send_json({"error": "notes is required -- describe what this run is"}, status=400)
            return

        if run_condition == "oracle_segmentation" and not oracle_rttm:
            self._send_json(
                {"error": "oracle_segmentation requires oracle_rttm (path to the only_words RTTM)"},
                status=400,
            )
            return

        extra_args = ["--refinement-strategy", refinement_strategy]
        if refinement_strategy == "oracle":
            extra_args += ["--oracle-scope", oracle_scope or "all_pairs"]
        if counts_toward_results:
            extra_args.append("--counts-toward-results")

        run_id = uuid.uuid4().hex[:12]
        with _lock:
            _runs[run_id] = {
                "run_id": run_id,
                "run_condition": run_condition,
                "notes": notes,
                "status": "starting",
                "total": 0,
                "done": 0,
            }

        thread = threading.Thread(
            target=_launch_run,
            args=(run_id, run_condition, notes, extra_args, data_root, condition, split, oracle_rttm),
            daemon=True,
        )
        thread.start()

        self._send_json({"run_id": run_id})

    def log_message(self, format: str, *args) -> None:  # noqa: A002 -- stdlib signature
        pass  # keep stdout clean; use /api/log to inspect a run's output


def main() -> None:
    server = ThreadingHTTPServer(("localhost", 8765), Handler)
    print("Harness UI running at http://localhost:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
