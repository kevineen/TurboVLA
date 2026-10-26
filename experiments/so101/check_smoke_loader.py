"""GR-BC wiring check: smoke dataset through the starVLA loader with include_index.

Run from ~/robot/TurboVLA with .venv310:
  PYTHONPATH=$PWD:$PWD/third_party/starvla_runtime .venv310/bin/python experiments/so101/check_smoke_loader.py
"""

from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset

cfg = OmegaConf.create(
    dict(
        data_root_dir="/home/kevin/robot/datasets",
        data_mix="grbc_smoke",
        dataset_sampling_strategy="default",
        include_state=True,
        include_index=True,
        obs_image_size=[224, 224],
        video_backend="pyav",
        load_all_data_for_training=True,
    )
)

ds = get_vla_dataset(data_cfg=cfg)
print("dataset length (steps):", len(ds))

seen = set()
for probe in (0, len(ds) // 2, len(ds) - 1):
    s = ds[probe]
    keys = sorted(s.keys())
    print(f"[{probe}] keys:", keys)
    assert "index" in s, "include_index=True must add 'index' to the sample"
    seen.add(int(s["index"]))
    print(f"[{probe}] index={s['index']} lang={s['lang']!r} action={s['action'].shape} state={s['state'].shape}")
    print(f"[{probe}] images:", [f"{type(im).__name__} {im.size}" for im in s["image"]])

# The index must be a dataset-global frame id: mixture of 480 frames -> ints in [0, 479]
assert all(0 <= i < 480 for i in seen), seen
print("distinct indices probed:", sorted(seen))
print("OK")
