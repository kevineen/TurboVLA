#!/usr/bin/env bash
set -euo pipefail

# Download the 50-task Clean and Randomized LeRobot datasets into one training root.
# Usage: bash scripts/robotwin/prepare_data.sh [ROBOTWIN_DATA_ROOT]
# The optional positional path (or DEST) is the final root, not its parent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if (( $# > 1 )); then
    echo "Usage: bash scripts/robotwin/prepare_data.sh [ROBOTWIN_DATA_ROOT]" >&2
    exit 2
fi

data_root="${1:-${ROBOTWIN_DATA_ROOT:-${DEST:-${REPO_ROOT}/playground/Datasets/RoboTwin}}}"
clean_repo="${ROBOTWIN_CLEAN_REPO:-StarVLA/RoboTwin-Clean}"
randomized_repo="${ROBOTWIN_RANDOMIZED_REPO:-StarVLA/RoboTwin-Randomized}"

command -v hf >/dev/null 2>&1 || {
    echo '[ERROR] Hugging Face CLI not found. Install huggingface-hub[hf_xet] first.' >&2
    exit 1
}

mkdir -p "${data_root}/Clean" "${data_root}/Randomized"

echo "[INFO] Downloading ${clean_repo} into ${data_root}/Clean"
# This repository has task directories at its root, not a top-level Clean/ folder.
hf download "${clean_repo}" --repo-type dataset --local-dir "${data_root}/Clean"

echo "[INFO] Downloading Randomized/ from ${randomized_repo} into ${data_root}"
# The Randomized repository also contains a Clean/ tree; exclude it so the two
# sources cannot overwrite each other's task files.
hf download "${randomized_repo}" --repo-type dataset \
    --include 'Randomized/**' --local-dir "${data_root}"

mapfile -t tasks < <(find "${data_root}/Clean" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%f\n' | sort)
if (( ${#tasks[@]} != 50 )); then
    echo "[ERROR] Expected 50 Clean task directories, found ${#tasks[@]} in ${data_root}/Clean" >&2
    exit 1
fi
for task in "${tasks[@]}"; do
    for variant in Clean Randomized; do
        task_dir="${data_root}/${variant}/${task}"
        for required in data meta videos; do
            if [[ ! -d "${task_dir}/${required}" ]]; then
                echo "[ERROR] Missing ${task_dir}/${required}" >&2
                exit 1
            fi
        done
    done
done

echo "[INFO] Ready: 50 Clean + 50 Randomized tasks under ${data_root}"
echo "export ROBOTWIN_DATA_ROOT='${data_root}'"
