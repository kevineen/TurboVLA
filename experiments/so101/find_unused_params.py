"""Diagnose which trainable parameters do not participate in the loss."""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ.setdefault("BERT_MODEL_PATH", "/home/kevin/models/bert-base-uncased")
os.environ.setdefault("DINOV3_MODEL_PATH", "/home/kevin/models/dinov3-vitb16")
os.environ.setdefault("TURBOVLA_INIT_CKPT", "/home/kevin/models/groundingdino/groundingdino_swint_ogc.pth")

from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework  # noqa: E402

cfg = OmegaConf.load("experiments/so101/configs/cobra_so101.yaml")
fw = TurboVLAFramework(config=OmegaConf.create({"framework": OmegaConf.to_container(cfg.framework)})).cuda()

from PIL import Image  # noqa: E402

example = {
    "image": [Image.new("RGB", (640, 480))] * int(cfg.framework.vision.num_views),
    "lang": "Grab the cobra by the neck and put its whole body inside the clear plastic container.",
    "state": np.zeros((1, cfg.framework.action.state_dim), dtype=np.float32),
    "action": np.zeros((cfg.framework.action.horizon, cfg.framework.action.action_dim), dtype=np.float32),
}
out = fw([example])
loss = out["action_loss"]
loss.backward()

unused = [n for n, p in fw.named_parameters() if p.requires_grad and p.grad is None]
print("loss:", float(loss))
print("trainable params:", sum(p.numel() for p in fw.parameters() if p.requires_grad))
print("unused trainable params:")
for n in unused:
    print("  ", n)
