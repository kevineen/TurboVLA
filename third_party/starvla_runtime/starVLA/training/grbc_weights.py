"""GR-BC precomputed graph weights for TurboVLA weighted SFT.

Vendored (with minor adaptation) from Gamma-Robot/GR-BC
`src/lerobot/policies/grbc/training.py` (Apache-2.0, CoRL 2026,
arXiv:2610.20504). The upstream module is LeRobot-coupled only through
`validate_dataset`; here coverage is checked against an explicit index
array and batch weights are computed from the starVLA batch format
(`List[dict]` with an opt-in `"index"` key, see `include_index`).

Weight semantics are unchanged from upstream:
- `index` is the dataset-global frame number of the action-chunk start.
- reduction="mean" divides by batch size (weighted mean),
  "self_normalized" divides by the weight sum.
- A DDP-corrected global denominator compensates gradient averaging.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class GRBCWeights:
    def __init__(self, path, column="graph_transition_weight", reduction="mean", device="cpu"):
        if reduction not in ("mean", "self_normalized"):
            raise ValueError("grbc reduction must be mean or self_normalized")
        table = pd.read_parquet(path)
        if {"index", column} - set(table):
            raise ValueError(f"Weights require index and {column}")
        indices = table["index"].to_numpy()
        if not np.isfinite(indices).all() or not np.equal(indices, indices.astype(np.int64)).all():
            raise ValueError("Weight indices must be finite integers")
        weights = table[column].to_numpy(dtype=np.float32)
        if table["index"].duplicated().any() or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Duplicate indices, nonfinite or negative weights")
        self.lookup = pd.Series(weights, index=indices.astype(np.int64))
        self.table, self.device, self.reduction = table, device, reduction
        summary = Path(path).with_name("summary.json")
        self.metadata = json.loads(summary.read_text()) if summary.exists() else None

    def validate_indices(self, indices):
        """Check that every trainable sample index has a weight (run once at startup)."""
        indices = np.asarray(indices, dtype=np.int64)
        if not np.isin(indices, self.lookup.index).all():
            missing = indices[~np.isin(indices, self.lookup.index)]
            raise ValueError(
                f"Weight table does not cover {len(missing)} dataset indices "
                f"(first missing: {missing[:10]}); regenerate weights for this dataset"
            )

    def validate_horizon(self, action_horizon):
        """Weights must have been exported with n_action_step == policy horizon."""
        expected = int(self.metadata["config"]["n_action_step"]) if self.metadata else None
        if expected is not None and expected != int(action_horizon):
            raise ValueError(
                f"GR-BC weights were exported with n_action_step={expected} but the policy "
                f"horizon is {action_horizon}; re-export with --n-action-step {action_horizon}"
            )

    def compute_batch_weights(self, examples):
        """Batch weights for a starVLA batch (`List[dict]`, identity collate)."""
        if not isinstance(examples, (list, tuple)) or not examples or "index" not in examples[0]:
            raise KeyError("GRBC requires the global sample index; set include_index=True in vla_data")
        ids = np.asarray([int(example["index"]) for example in examples], dtype=np.int64)
        selected = self.lookup.reindex(ids)
        if selected.isna().any():
            raise KeyError(f"No weight for indices: {ids[selected.isna().to_numpy()][:10]}")
        weights = torch.tensor(selected.to_numpy(), dtype=torch.float32, device=self.device)
        return weights, {
            "grbc_mean_weight": weights.mean().item(),
            "grbc_zero_fraction": (weights == 0).float().mean().item(),
        }


def weighted_loss(per_sample, weights, accelerator=None, reduction="mean"):
    if per_sample.ndim != 1 or per_sample.shape != weights.shape:
        raise ValueError(f"Expected matching (B,) losses/weights, got {per_sample.shape}, {weights.shape}")
    if reduction not in ("mean", "self_normalized"):
        raise ValueError("Unknown GRBC reduction")
    weights = weights.to(device=per_sample.device, dtype=torch.float32)
    denominator = weights.sum() if reduction == "self_normalized" else weights.new_tensor(weights.numel())
    world_size = accelerator.num_processes if accelerator is not None else 1
    if accelerator is not None:
        denominator = accelerator.reduce(denominator.detach(), reduction="sum")
    # DDP averages local gradients, so compensate once for the global denominator.
    return (per_sample.float() * weights).sum() * world_size / denominator.clamp_min(1e-12)
