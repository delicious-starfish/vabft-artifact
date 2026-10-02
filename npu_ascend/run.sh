#!/usr/bin/env bash

# Full V-ABFT exponent-bit experiment followed by XLSX/Markdown parsing.
#
# Usage:
#   bash npu_ascend/run.sh [OUTPUT_DIR] [extra experiment arguments]
#
# Examples:
#   bash npu_ascend/run.sh
#   DEVICE=npu:1 bash npu_ascend/run.sh results/my_full_run
#   ENABLE_AUDIT=1 bash npu_ascend/run.sh results/my_audited_run
#
# ENABLE_AUDIT is disabled by default because a full run can produce tens of
# millions of JSONL records and consume many GB of disk space.

source ~/.bashrc
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT_DIR="$1"
    shift
else
    OUTPUT_DIR="results/vabft_acc_hiPre_full_$(date +%Y%m%d_%H%M%S)"
fi

DEVICE="${DEVICE:-npu:0}"
TRIALS="${TRIALS:-10000}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-256}"
P_INJECT="${P_INJECT:-0.5}"
ENABLE_AUDIT="${ENABLE_AUDIT:-0}"

mkdir -p "${OUTPUT_DIR}"

RESULT_CSV="${OUTPUT_DIR}/results.csv"
REPORT_XLSX="${OUTPUT_DIR}/report.xlsx"
REPORT_MD="${OUTPUT_DIR}/report.md"
RUN_LOG="${OUTPUT_DIR}/run.log"
PARSE_LOG="${OUTPUT_DIR}/parse.log"
AUDIT_JSONL="${OUTPUT_DIR}/audit.jsonl"

experiment_cmd=(
    python "${SCRIPT_DIR}/vabft_acc_hiPre.py"
    --m 1024
    --n 1024
    --k-values 1024,2048,3072,4096,6144,8192
    --dtypes bf16,fp32
    --trials "${TRIALS}"
    --refresh-interval "${REFRESH_INTERVAL}"
    --p-inject "${P_INJECT}"
    --device "${DEVICE}"
    --output "${RESULT_CSV}"
)

parse_cmd=(
    python "${SCRIPT_DIR}/parse_vabft_acc_hiPre.py"
    --input "${RESULT_CSV}"
    --xlsx "${REPORT_XLSX}"
    --markdown "${REPORT_MD}"
)

if [[ "${ENABLE_AUDIT}" == "1" ]]; then
    experiment_cmd+=(--audit-jsonl "${AUDIT_JSONL}")
    parse_cmd+=(--audit "${AUDIT_JSONL}")
fi

# Optional arguments make it possible to launch a smoke run through the same
# pipeline.  Later CLI arguments override earlier argparse values.
experiment_cmd+=("$@")

echo "Output directory: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Trials per condition: ${TRIALS}"
echo "Matrix refresh interval: ${REFRESH_INTERVAL}"
echo "Audit enabled: ${ENABLE_AUDIT}"
echo "Starting experiment..."

PYTHONUNBUFFERED=1 "${experiment_cmd[@]}" 2>&1 | tee "${RUN_LOG}"

echo "Experiment complete. Generating reports..."
PYTHONUNBUFFERED=1 "${parse_cmd[@]}" 2>&1 | tee "${PARSE_LOG}"

echo "Pipeline complete."
echo "  CSV:      ${RESULT_CSV}"
echo "  XLSX:     ${REPORT_XLSX}"
echo "  Markdown: ${REPORT_MD}"
echo "  Run log:  ${RUN_LOG}"
if [[ "${ENABLE_AUDIT}" == "1" ]]; then
    echo "  Audit:    ${AUDIT_JSONL}"
fi
