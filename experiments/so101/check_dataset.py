"""Dry-run the cobra dataset through the starVLA GR00T loader (no GPU/model).

Verifies: mixture discovery, modality.json resolution, language retrieval,
state/action slicing + min_max normalization, image decode/resize.
"""

from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset

cfg = OmegaConf.create(
    dict(
        data_root_dir="/home/kevin/.cache/huggingface/lerobot/local",
        data_mix="cobra",
        dataset_sampling_strategy="default",
        include_state=True,
        obs_image_size=[224, 224],
        video_backend="pyav",
        load_all_data_for_training=True,
    )
)

ds = get_vla_dataset(data_cfg=cfg)
print("dataset length (steps):", len(ds))

s = ds[0]
print("keys:", sorted(s.keys()))
print("lang:", repr(s["lang"]))
print("action:", s["action"].shape, "dtype", s["action"].dtype,
      "min", float(s["action"].min()), "max", float(s["action"].max()))
print("state:", s["state"].shape, s["state"].round(decimals=3).tolist())
print("images:", [f"{type(im).__name__} {im.size}" for im in s["image"]])

s2 = ds[len(ds) // 2]
print("mid lang:", repr(s2["lang"]), "| action min/max:",
      float(s2["action"].min()), float(s2["action"].max()))
print("OK")
