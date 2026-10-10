#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${REPO_ROOT}/third_party/starvla_runtime"
export PYTHONPATH="${REPO_ROOT}:${RUNTIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export BERT_MODEL_PATH="${BERT_MODEL_PATH:-${TURBOVLA_TEXT_ENCODER_PATH:-}}"
export TURBOVLA_INIT_CKPT="${TURBOVLA_INIT_CKPT:-${GROUNDINGDINO_CKPT:-}}"

: "${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT to the converted RoboTwin dataset root.}"
: "${BERT_MODEL_PATH:?Set BERT_MODEL_PATH to a local bert-base-uncased directory.}"
: "${DINOV3_MODEL_PATH:?Set DINOV3_MODEL_PATH to a local DINOv3 model directory.}"
: "${TURBOVLA_INIT_CKPT:?Set TURBOVLA_INIT_CKPT to the GroundingDINO checkpoint.}"

config_yaml="${CONFIG_YAML:-${REPO_ROOT}/experiments/robotwin/configs/taskbalanced_all50.yaml}"
run_root_dir="${RUN_ROOT_DIR:-${REPO_ROOT}/results/Checkpoints}"
gpus_per_node="${GPUS_PER_NODE:-${RESOURCE_GPU:-8}}"
num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
num_processes="${NUM_PROCESSES:-$((gpus_per_node * num_machines))}"
main_process_ip="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
main_process_port="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29630}}"
launcher_python="${STARVLA_PYTHON:-python}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-64}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
max_train_steps="${MAX_TRAIN_STEPS:-150000}"
warmup_steps="${WARMUP_STEPS:-1000}"
save_interval="${SAVE_INTERVAL:-10000}"
logging_frequency="${LOGGING_FREQUENCY:-50}"
learning_rate="${LEARNING_RATE:-5.0e-5}"
image_size="${IMAGE_SIZE:-224}"
randomized_to_clean_ratio="${RANDOMIZED_TO_CLEAN_RATIO:-10.0}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"
is_resume="${IS_RESUME:-false}"

case "${is_resume,,}" in
    1|true|yes|on) is_resume=true ;;
    0|false|no|off) is_resume=false ;;
    *) echo "[ERROR] IS_RESUME must be true or false" >&2; exit 2 ;;
esac

if [[ -n "${RUN_ID:-}" ]]; then
    run_id="${RUN_ID}"
elif [[ "${num_machines}" -eq 1 && "${is_resume}" == false ]]; then
    run_id="turbovla_robotwin_all50_taskbalanced_r10_gbs512_$(date +%Y%m%d_%H%M%S)"
else
    echo "[ERROR] Multi-machine runs and resumes require a shared RUN_ID" >&2
    exit 2
fi
output_dir="${run_root_dir}/${run_id}"

for required in "${config_yaml}" "${ROBOTWIN_DATA_ROOT}/Clean" "${ROBOTWIN_DATA_ROOT}/Randomized" "${BERT_MODEL_PATH}/config.json" "${DINOV3_MODEL_PATH}/config.json" "${TURBOVLA_INIT_CKPT}"; do
    [[ -e "${required}" ]] || { echo "[ERROR] Required path missing: ${required}" >&2; exit 1; }
done
if [[ "${launcher_python}" == */* ]]; then
    [[ -x "${launcher_python}" ]] || { echo "[ERROR] Python not executable: ${launcher_python}" >&2; exit 1; }
fi
if ! "${launcher_python}" -c "import accelerate, omegaconf, pydantic, starVLA" >/dev/null; then
    echo "[ERROR] RoboTwin training dependency preflight failed" >&2
    exit 1
fi
if [[ "${machine_rank}" -eq 0 ]]; then
    if [[ "${is_resume}" == true && ! -d "${output_dir}/checkpoints" ]]; then
        echo "[ERROR] Resume checkpoint directory missing: ${output_dir}/checkpoints" >&2
        exit 1
    fi
    if [[ "${is_resume}" == false && -e "${output_dir}" ]]; then
        echo "[ERROR] Output directory already exists: ${output_dir}" >&2
        exit 1
    fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,veth}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

echo "[INFO] TurboVLA RoboTwin all50: 50 tasks, Clean:Randomized=1:${randomized_to_clean_ratio}"
echo "[INFO] output_dir=${output_dir} resume=${is_resume}"
echo "[INFO] GPUs=${num_processes} per_device_bs=${per_device_batch_size} grad_accum=${gradient_accumulation_steps} global_bs=$((num_processes * per_device_batch_size * gradient_accumulation_steps))"
echo "[INFO] image_size=${image_size} max_steps=${max_train_steps} warmup=${warmup_steps} lr=${learning_rate}"

"${launcher_python}" -m accelerate.commands.launch \
    --config_file "${REPO_ROOT}/experiments/robotwin/configs/deepspeed_zero2.yaml" \
    --num_processes "${num_processes}" \
    --num_machines "${num_machines}" \
    --machine_rank "${machine_rank}" \
    --main_process_ip "${main_process_ip}" \
    --main_process_port "${main_process_port}" \
    "${RUNTIME_ROOT}/starVLA/training/train_turbovla.py" \
    --config_yaml "${config_yaml}" \
    --datasets.vla_data.data_root_dir "${ROBOTWIN_DATA_ROOT}" \
    --datasets.vla_data.data_mix robotwin_all_50 \
    --datasets.vla_data.dataset_sampling_strategy task_uniform_variant_ratio \
    --datasets.vla_data.randomized_to_clean_ratio "${randomized_to_clean_ratio}" \
    --datasets.vla_data.obs_image_size "[${image_size},${image_size}]" \
    --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
    --framework.vision.image_size "${image_size}" \
    --trainer.learning_rate.base "${learning_rate}" \
    --trainer.learning_rate.text_encoder "${learning_rate}" \
    --trainer.learning_rate.vision_encoder "${learning_rate}" \
    --trainer.learning_rate.vision_language_interaction "${learning_rate}" \
    --trainer.learning_rate.vision_projection "${learning_rate}" \
    --trainer.learning_rate.action_head "${learning_rate}" \
    --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --trainer.is_resume "${is_resume}" \
    --trainer.ema_decay "${ema_decay}" \
    --trainer.ema_device "${ema_device}" \
    --trainer.max_train_steps "${max_train_steps}" \
    --trainer.num_warmup_steps "${warmup_steps}" \
    --trainer.save_interval "${save_interval}" \
    --trainer.logging_frequency "${logging_frequency}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project "${WANDB_PROJECT:-TurboVLA_Robotwin}" \
    --wandb_entity "${WANDB_ENTITY:-your_wandb_entity}" \
    "$@"
