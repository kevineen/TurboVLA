"""SO-101 single-arm (cobra task) data configuration for TurboVLA.

Auto-discovered by starVLA.dataloader.gr00t_lerobot.registry. Dataset is a
stock LeRobot v3.0 dataset; column mapping lives in the dataset's
meta/modality.json (original_key points at standard LeRobot columns).
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class So101DataConfig:
    """Two-camera SO-101 single-arm data, 6-D absolute joint actions (deg)."""

    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.arm"]
    action_keys = ["action.arm"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(12))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # 全次元が deg（グリッパー含む）なので binary は使わない
        state_modes = {key: "min_max" for key in self.state_keys}
        action_modes = {key: "min_max" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"so101": So101DataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
DATASET_NAMED_MIXTURES = {
    # Legacy plastic-container curated set (archived; do not use for new evals).
    "cobra": [
        ("so101_singlearm_pickup_cobra_20261018_205119_curated", 1.0, "so101"),
    ],
    # Current task: silver bowl (2026-10-20 RGB front+wrist).
    "cobra_silver_bowl": [
        ("so101_singlearm_pickup_cobra_rgb_20261020_225627", 1.0, "so101"),
    ],
    # GR-BC wiring smoke (synthetic 2-view pool under ~/robot/datasets; disposable).
    "grbc_smoke": [
        ("grbc_smoke_v3", 1.0, "so101"),
    ],
}
