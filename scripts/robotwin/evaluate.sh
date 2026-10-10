#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    echo "Usage: bash scripts/robotwin/evaluate.sh <checkpoint> [--mode both|clean|randomized] [task ...]" >&2
}

ckpt_path="${1:-${CKPT_PATH:-}}"
if [[ -z "${ckpt_path}" ]]; then
    usage
    exit 1
fi
shift || true

eval_mode="${ROBOTWIN_EVAL_MODE:-both}"
if [[ "${1:-}" == "--mode" ]]; then
    [[ $# -ge 2 ]] || { echo "[ERROR] --mode requires a value" >&2; exit 2; }
    eval_mode="$2"
    shift 2
elif [[ "${1:-}" == --mode=* ]]; then
    eval_mode="${1#--mode=}"
    shift
fi
case "${eval_mode}" in
    both) variants=(demo_clean demo_randomized) ;;
    clean|demo_clean) variants=(demo_clean) ;;
    randomized|demo_randomized) variants=(demo_randomized) ;;
    *) echo "[ERROR] Unsupported evaluation mode: ${eval_mode}" >&2; usage; exit 2 ;;
esac

if [[ ! -f "${ckpt_path}" ]]; then
    echo "[ERROR] Checkpoint not found: ${ckpt_path}" >&2
    exit 1
fi

: "${ROBOTWIN_PATH:?Set ROBOTWIN_PATH to the RoboTwin repository root.}"

export STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
export ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
export DEPLOY_POLICY_TEMPLATE_PATH="${DEPLOY_POLICY_TEMPLATE_PATH:-${REPO_ROOT}/experiments/robotwin/evaluation/deploy_policy.yml}"

tasks=("$@")
if (( ${#tasks[@]} == 0 )); then
    tasks=(all)
fi

policy_base="${POLICY_NAME:-turbovla_all50}"
for variant in "${variants[@]}"; do
    variant_name="${variant#demo_}"
    bash "${SCRIPT_DIR}/start_eval.sh" \
        --mode "${variant}" \
        --name "${policy_base}_${variant_name}" \
        --ckpt "${ckpt_path}" \
        --jobs-per-gpu "${ROBOTWIN_JOBS_PER_GPU:-1}" \
        --base-port "${ROBOTWIN_BASE_PORT:-7100}" \
        "${tasks[@]}"
done
