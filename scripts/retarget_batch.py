"""Run independent WholeBody V5 jobs without allowing one failure to abort the batch."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="complete motion input; repeat for multiple inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--tgt-fps", type=float, default=50.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    entry = Path(__file__).with_name("retarget.py")
    for motion in args.input:
        motion = motion.expanduser().resolve()
        output = args.output_dir / f"{motion.stem}.pkl"
        command = [sys.executable, str(entry), "--motion", str(motion),
                   "--robot", args.robot, "--output", str(output),
                   "--tgt_fps", str(args.tgt_fps)]
        if args.config is not None:
            command += ["--config", str(args.config.expanduser().resolve())]
        completed = subprocess.run(command, text=True, capture_output=True)
        status_path = output.with_suffix(".status.json")
        status = "VALID" if completed.returncode == 0 and output.is_file() else "FAILED"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text()).get("status", status)
            except Exception:
                pass
        records.append({"input": str(motion), "output": str(output),
                        "status": status, "returncode": completed.returncode,
                        "stdout": completed.stdout[-2000:],
                        "stderr": completed.stderr[-2000:]})
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    report = {"frames_are_not_truncated": True, "counts": counts, "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": counts, "report": str(args.report.resolve())}, ensure_ascii=False))
    return 0 if all(item["status"] == "VALID" for item in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
