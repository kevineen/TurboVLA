# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from collections import defaultdict

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    DATASET_NAMED_MIXTURES,
    EmbodimentTag,
)

def collate_fn(batch):
    return batch


def _task_balanced_robotwin_weights(entries, randomized_to_clean_ratio: float):
    """Give each task equal mass, then split its Clean/Randomized mass 1:r."""
    if randomized_to_clean_ratio <= 0:
        raise ValueError("randomized_to_clean_ratio must be positive")
    by_task = defaultdict(dict)
    for dataset, source_weight, name in entries:
        variant, separator, task = name.partition("/")
        if not separator or variant not in {"Clean", "Randomized"}:
            raise ValueError(f"Expected a Clean/<task> or Randomized/<task> dataset, got {name!r}")
        if variant in by_task[task]:
            raise ValueError(f"Duplicate {variant} dataset for task {task!r}")
        if source_weight <= 0 or len(dataset) == 0:
            raise ValueError(f"Dataset {name!r} must have a positive weight and non-empty data")
        by_task[task][variant] = dataset

    if len(by_task) != 50 or any(set(variants) != {"Clean", "Randomized"} for variants in by_task.values()):
        raise ValueError("Task-balanced RoboTwin all50 requires both variants of all 50 tasks")

    task_mass = 1.0 / len(by_task)
    clean_mass = task_mass / (1.0 + randomized_to_clean_ratio)
    randomized_mass = task_mass - clean_mass
    print(
        "[INFO] task-balanced RoboTwin sampling: "
        f"tasks={len(by_task)} Clean:Randomized=1:{randomized_to_clean_ratio:g}"
    )
    return [
        (by_task[task][variant], clean_mass if variant == "Clean" else randomized_mass)
        for task in sorted(by_task)
        for variant in ("Clean", "Randomized")
    ]

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = True,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    named_datasets = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )
        dataset_mixture.append((dataset, d_weight))
        named_datasets.append((dataset, d_weight, d_name))

    sampling_strategy = str(data_cfg.get("dataset_sampling_strategy", "default"))
    if sampling_strategy == "task_uniform_variant_ratio":
        dataset_mixture = _task_balanced_robotwin_weights(
            named_datasets,
            float(data_cfg.get("randomized_to_clean_ratio", 10.0)),
        )
        balance_dataset_weights = False
        balance_trajectory_weights = False
    elif sampling_strategy != "default":
        raise ValueError(f"Unsupported dataset_sampling_strategy: {sampling_strategy}")

    print(
        "[INFO] LeRobotMixtureDataset "
        f"balance_dataset_weights={balance_dataset_weights} "
        f"balance_trajectory_weights={balance_trajectory_weights}"
    )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
