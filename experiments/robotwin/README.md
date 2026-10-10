# RoboTwin 2.0 all50

This directory contains the RoboTwin 2.0 training and evaluation entry points. RoboTwin and LIBERO use the existing shared model in `../../turbovla/models/`.

The model uses three DINOv3 camera views, online BERT instruction encoding, bidirectional vision-language interaction, a 50-step ACT head, and 14-D bimanual actions.

## Layout

```text
configs/                         all50 recipe, modality, and DeepSpeed configuration
data_registry/                   Clean/Randomized 50-task mixtures and embodiment contract
evaluation/                      policy client and simulator adapter
train.py                         Python training entry point
evaluate.py                      Python evaluation entry point
../../scripts/robotwin/          data, training, serving, and evaluation scripts
../../third_party/               StarVLA-compatible runtime and licenses
```

## Installation

Python 3.10 or newer is required.

```bash
pip install -e ".[robotwin]"
```

Install a CUDA-compatible PyTorch build and FlashAttention 2 separately when needed by the selected environment.

## Required assets

```bash
export ROBOTWIN_DATA_ROOT=/path/to/converted/RoboTwin
export BERT_MODEL_PATH=/path/to/bert-base-uncased
export TURBOVLA_INIT_CKPT=/path/to/groundingdino_swint_ogc.pth
export DINOV3_MODEL_PATH=/path/to/dinov3
```

`ROBOTWIN_DATA_ROOT` must contain the same 50 tasks under both `Clean/<task_name>` and `Randomized/<task_name>`, in the converted LeRobot layout. Run `bash scripts/robotwin/prepare_data.sh /path/to/converted/RoboTwin` after installing the Hugging Face `hf` CLI; the script downloads both variants and checks the layout.

## Training

The RoboTwin recipe reported in the paper is defined in `taskbalanced_all50.yaml`: eight GPUs at 64 samples per GPU (global batch 512), 150k optimizer steps, learning rate `5e-5`, 1k warmup steps, AdamW weight decay `1e-10`, and EMA decay `0.999`. It samples the 50 tasks uniformly, Clean:Randomized at 1:10 within each task, then trajectories and start indices uniformly. Images are resized to 224×224. The architecture is frozen BERT, trainable DINOv3, six-layer feature enhancer, and a 50-action ACT head.

```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/robotwin/train.sh
```

Override `NUM_PROCESSES`, `PER_DEVICE_BATCH_SIZE`, `MAX_TRAIN_STEPS`, `LEARNING_RATE`, `RUN_ROOT_DIR`, or `RUN_ID` as needed. Multi-node runs and resume require a shared `RUN_ID`; for resume, set `IS_RESUME=true`.

## Evaluation

Install RoboTwin separately and set:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=/path/to/policy-env/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
```

Evaluate all 50 tasks in both Clean and Randomized, with 100 episodes per task and variant:

```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/robotwin/evaluate.sh \
  /path/to/steps_150000_ema_pytorch_model.pt
```

Pass `--mode clean` or `--mode randomized` to run one variant, and append task names to evaluate a subset. `ROBOTWIN_JOBS_PER_GPU` controls per-GPU concurrency.

Use [RoboTwin 2.0 commit `bf44be5`](https://github.com/RoboTwin-Platform/RoboTwin/commit/bf44be5)
for evaluation; the adapter is not guaranteed to work with newer simulator
revisions. At this commit, `script/eval_policy.py` runs 100 episodes per task
and does not read `ROBOTWIN_TEST_NUM`.

For checkpoints whose configuration uses `framework.dinov3`, `framework.fusion`, and `framework.action_model` keys, the RoboTwin adapter translates the configuration and state-dict keys to the shared model layout. Keep the corresponding `config.yaml` and `dataset_statistics.json` with the checkpoint, and verify numerical parity on a GPU before reporting evaluation results.

The compatibility runtime retains the `starVLA` and `deployment` package names used by existing checkpoints. See `third_party/licenses/StarVLA-MIT.txt`, `third_party/licenses/Apache-2.0.txt`, and the source-file copyright notices.
