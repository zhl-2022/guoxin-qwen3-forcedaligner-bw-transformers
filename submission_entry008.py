"""Reproducible offline development entrypoints; no official-score fabrication.

Copyright (c) 2026 the task author. Existing upstream licenses are unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TASK = ROOT / "tasks/qwen3-forcedaligner-bw-transformers"
PACKAGES = ("torch", "transformers", "qwen-asr", "accelerate", "librosa", "soundfile", "nagisa", "soynlp", "vllm", "vllm-hcu", "triton")


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def snapshot(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("run", "env", "smoke", "verify", "bench"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--model-dir", default=os.environ.get("FORCEDALIGNER_MODEL_PATH", "/root/private_data/guoxin-candidates/models/forcedaligner"))
    ap.add_argument("--upstream-root", default=os.environ.get("FORCEDALIGNER_UPSTREAM_ROOT", "/root/private_data/guoxin-candidates/source/qwen3-asr"))
    ap.add_argument("--requests", type=Path, default=TASK / "requests.example.json")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--result-dir", type=Path)
    ap.add_argument("--max-seconds", type=int, default=360)
    args = ap.parse_args(argv)
    if args.selfcheck and args.action != "verify":
        ap.error("--selfcheck is only meaningful for the official verify entry")
    if args.action in ("verify", "bench"):
        emit({"status": "official_materials_and_contract_pending", "action": args.action,
              "selfcheck_requested": args.selfcheck, "executed": False,
              "missing": ["official ForcedAligner evaluator/selfcheck", "fixed audio/text and independent timestamp reference",
                          "official GPU baseline and AAS aggregation/tolerance contract", "task-specific benchmark load and measurement rules"],
              "framework": "publisher_transformers_SDPA", "framework_acceptance_confirmed": False,
              "official_accuracy_verified": False, "official_performance_verified": False,
              "submission_ready": False, "result_file_created": False})
        return 2
    if args.action == "env":
        packages = {}
        for name in PACKAGES:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result = {"UTC": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version,
                  "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                  "packages": packages, "model_id": "Qwen/Qwen3-ForcedAligner-0.6B",
                  "framework": "publisher_transformers_SDPA", "precision_configured": "bfloat16",
                  "image_reference_supplied": os.environ.get("IMAGE") or None,
                  "gpu_device_probe_executed": False, "metadata_only": True,
                  "note": "This metadata record is not a GPU device or clean-image replay proof."}
        if args.output and not args.dry_run:
            with args.output.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write("\n")
        emit(result)
        return 0
    if args.action == "run":
        if not 60 <= args.max_seconds <= 1800:
            ap.error("--max-seconds must be 60..1800")
        if args.output is None or not args.output.is_absolute():
            ap.error("run requires --output NEW_ABSOLUTE_DIRECTORY, including --dry-run")
        if args.output.exists() or args.output.is_symlink():
            ap.error("run output already exists; preserving it")
        command = [sys.executable, "-B", str(TASK / "entry.py"), "run", "--model-dir", args.model_dir,
                   "--upstream-root", args.upstream_root, "--requests", str(args.requests.resolve()),
                   "--output", str(args.output), "--max-seconds", str(args.max_seconds)]
        if args.dry_run:
            emit({"command": command, "GPU_started": False, "filesystem_modified": False,
                  "scope": "Same frozen offline Transformer aligner; not a native vLLM/SGLang route"})
            return 0
        if sys.platform != "linux":
            ap.error("actual execution requires the prepared SCNet Linux environment")
        # The frozen canonical entry validates weights/source and owns/reaps its
        # model process group with its own explicit deadline.
        return subprocess.call(command)
    if args.result_dir is None:
        ap.error("smoke requires --result-dir DIRECTORY_FROM_COMPLETED_RUN")
    worker = args.result_dir / "worker-result.json"
    final = args.result_dir / "run-final.json"
    if args.dry_run:
        emit({"worker_result": str(worker), "run_final": str(final), "GPU_started": False,
              "result_checked": False, "official_accuracy_verified": False})
        return 0
    result = json.loads(worker.read_text(encoding="utf-8"))
    lifecycle = json.loads(final.read_text(encoding="utf-8"))
    if (result.get("mode") != "publisher_transformers_sdpa_alignment"
            or result.get("development_contract_passed") is not True
            or lifecycle.get("status") != "completed" or lifecycle.get("worker_exit_code") != 0):
        raise ValueError("Result does not describe a completed offline alignment run")
    sys.path.insert(0, str(ROOT / "scripts"))
    from task_expansion_candidates_20261023 import validate_alignment
    records = result.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError("Empty alignment result set")
    checks = [validate_alignment(r["items"], r["input"]["text"], r["fixture"]["duration_seconds"]) for r in records]
    emit({"status": "development_artifact_contract_passed", "checked": checks,
          "inputs": [snapshot(worker), snapshot(final)], "new_GPU_execution": False,
          "official_accuracy_verified": False, "official_performance_verified": False})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Validation failed: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
