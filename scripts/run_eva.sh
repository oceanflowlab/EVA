#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PRED_FORMAT=tsv
DATA="${DATA:-HallusionBench}"
read -r -a DATASETS <<< "$DATA"
ARGS=(--backbone "${BACKBONE:-qwen}" --data "${DATASETS[@]}" --sample-n "${SAMPLE_N:-0}" --sample-seed "${SAMPLE_SEED:-42}")
if [[ -n "${WORK_DIR:-}" ]]; then ARGS+=(--work-dir "$WORK_DIR"); fi
if [[ "${REUSE:-0}" == "1" ]]; then ARGS+=(--reuse); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then ARGS+=(--dry-run); fi
exec python -m eva.evaluate "${ARGS[@]}" "$@"
