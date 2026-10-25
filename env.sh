#!/usr/bin/env bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID="Qwen/Qwen3-ForcedAligner-0.6B" CARD="BW" FRAMEWORK="publisher_transformers_SDPA" PRECISION="bf16"
export FORCEDALIGNER_PYTHON="${FORCEDALIGNER_PYTHON:-/root/private_data/guoxin-candidates/envs/forcedaligner-001/bin/python}"
export FORCEDALIGNER_MODEL_PATH="${FORCEDALIGNER_MODEL_PATH:-/root/private_data/guoxin-candidates/models/forcedaligner}"
export FORCEDALIGNER_UPSTREAM_ROOT="${FORCEDALIGNER_UPSTREAM_ROOT:-/root/private_data/guoxin-candidates/source/qwen3-asr}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
