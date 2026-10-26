#!/usr/bin/env python
"""GR-BC × TurboVLA 配線検証用の合成 SO-101 風データセット（破棄前提）。

仕様（本番データ契約に合わせる）:
  - カメラ2視点: observation.images.front / observation.images.wrist
  - state/action 6次元（SO-101 の 6関節）。末尾 = gripper.pos 生値 (0-52)
  - next.reward は成功エピソード最終フレームのみ 1.0
  - complementary_info.is_intervention に介入区間フラグ
  - meta/modality.json は TurboVLA(GR00Tローダ) 用に別途書き出し

使い方 (robot/GR-BC の .venv311):
  ~/robot/GR-BC/.venv311/bin/python make_smoke_dataset.py --output ~/robot/datasets/grbc_smoke_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset

RES = 256
FPS = 15
TASK = "pick up the cube and place it in the tray"
N_STATE = 6  # SO-101: 5 joints + gripper(raw 0-52)
GRIPPER_OPEN, GRIPPER_CLOSED = 45.0, 8.0


def lerp(a, b, t):
    return a + (b - a) * t


def render_front(obj, goal, gripper_closed, rng):
    """広視点カメラ（全局図）。"""
    img = Image.new("RGB", (RES, RES), (30, 30, 36))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, RES - 20, RES - 20], outline=(70, 70, 80), width=2)
    gx, gy = goal[0] * RES, goal[1] * RES
    d.rectangle([gx - 20, gy - 20, gx + 20, gy + 20], outline=(60, 200, 90), width=3)
    ox, oy = obj[0] * RES, obj[1] * RES
    s = 13
    d.rectangle([ox - s, oy - s, ox + s, oy + s], fill=(240, 90, 70) if not gripper_closed else (255, 160, 60))
    gap = 5 if gripper_closed else 17
    d.line([ox - s - gap, oy - s - 28, ox - s - gap, oy + s], fill=(200, 200, 210), width=4)
    d.line([ox + s + gap, oy - s - 28, ox + s + gap, oy + s], fill=(200, 200, 210), width=4)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 2.5, (RES, RES, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_wrist(obj, gripper_closed, rng):
    """手首カメラ（物体周辺のクローズアップ）。"""
    img = Image.new("RGB", (RES, RES), (24, 26, 30))
    d = ImageDraw.Draw(img)
    cx, cy = RES / 2, RES / 2
    ox, oy = cx + (obj[0] - 0.5) * 60, cy + (obj[1] - 0.5) * 60  # 物体は中心近傍で小さく動く
    s = 22
    d.rectangle([ox - s, oy - s, ox + s, oy + s], fill=(240, 90, 70) if not gripper_closed else (255, 160, 60))
    gap = 12 if gripper_closed else 38
    d.line([ox - s - gap, oy - s - 60, ox - s - gap, oy + s], fill=(210, 210, 220), width=8)
    d.line([ox + s + gap, oy - s - 60, ox + s + gap, oy + s], fill=(210, 210, 220), width=8)
    d.line([0, 40, RES, 40], fill=(90, 60, 60), width=10)  # アーム影
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 3.0, (RES, RES, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def episode_plan(kind: str, n: int, rng):
    start = (rng.uniform(0.10, 0.25), rng.uniform(0.12, 0.30))
    goal = (0.80, 0.76)
    t = np.linspace(0.0, 1.0, n)
    prog = np.clip((t - 0.05) / 0.85, 0, 1) ** 0.9
    if kind == "fail_stall":
        prog = np.clip(prog, 0, 0.4)
    if kind == "fail_drift":
        prog = np.clip(prog, 0, 0.7)
    if kind == "intervention":
        prog = np.where(t < 0.35, -0.25 * t, lerp(-0.0875, 1.0, (t - 0.35) / 0.65))

    objs, intervention = [], np.zeros(n, dtype=bool)
    for i in range(n):
        drift = (rng.uniform(-1, 1) * 0.012, rng.uniform(-1, 1) * 0.012)
        if kind == "fail_drift":
            drift = (drift[0] + 0.15 * t[i], drift[1] - 0.1 * t[i])
        if kind == "fail_stall":
            drift = (drift[0] * 3, drift[1] * 3)
        p = prog[i]
        objs.append((np.clip(lerp(start[0], goal[0], p) + drift[0], 0.02, 0.98),
                     np.clip(lerp(start[1], goal[1], p) + drift[1], 0.02, 0.98)))
        if kind == "intervention" and 0.35 <= t[i] < 0.80:
            intervention[i] = True
    gripper = np.where(prog > 0.55, GRIPPER_CLOSED, GRIPPER_OPEN)  # 進捗55%以降で把持
    return objs, gripper, intervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists")

    features = {
        "observation.state": {"dtype": "float32", "shape": (N_STATE,), "names": None},
        "action": {"dtype": "float32", "shape": (N_STATE,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "observation.images.front": {"dtype": "video", "shape": (3, RES, RES), "names": ["channels", "height", "width"]},
        "observation.images.wrist": {"dtype": "video", "shape": (3, RES, RES), "names": ["channels", "height", "width"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/grbc_smoke_v3", fps=FPS, features=features, root=str(out), use_videos=True
    )

    rng = np.random.default_rng(args.seed)
    kinds = (["success"] * max(1, args.episodes // 2) + ["fail_stall", "fail_drift", "intervention", "intervention"])[: args.episodes]

    for ep, kind in enumerate(kinds):
        objs, gripper, intervention = episode_plan(kind, args.frames, rng)
        success = kind in ("success", "intervention")
        prev = None
        for i in range(args.frames):
            state = np.zeros(N_STATE, dtype=np.float32)
            # 5関節: 物体位置から生成した準静的関節角
            state[0] = np.clip(objs[i][0] * 120 - 30, -90, 90)
            state[1] = np.clip(objs[i][1] * 100 - 40, -90, 90)
            state[2] = 40 * np.sin(i * 0.08 + ep)
            state[3] = 20 * np.cos(i * 0.06 + ep)
            state[4] = np.clip(90 - objs[i][0] * 60, -90, 90)
            state[5] = gripper[i]  # 末尾 = gripper.pos 生値 (0-52)
            action = state if prev is None else state - prev
            prev = state
            frame = {
                "observation.state": state,
                "action": action.astype(np.float32),
                "next.reward": np.array([1.0 if (success and i == args.frames - 1) else 0.0], dtype=np.float32),
                "next.done": np.array([i == args.frames - 1], dtype=np.bool_),
                "complementary_info.is_intervention": np.array([bool(intervention[i])], dtype=np.bool_),
                "task": TASK,
            }
            frng = np.random.default_rng(args.seed * 1009 + ep * 31 + i)
            frame["observation.images.front"] = render_front(objs[i], (0.80, 0.76), gripper[i] < 26, frng)
            frame["observation.images.wrist"] = render_wrist(objs[i], gripper[i] < 26, frng)
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"episode {ep}: {kind} frames={args.frames} success={success}")

    if dataset.image_writer is not None:
        dataset.stop_image_writer()

    modality = {
        "action": {"arm": {"start": 0, "end": 6, "original_key": "action"}},
        "state": {"arm": {"start": 0, "end": 6, "original_key": "observation.state"}},
        "video": {
            "front": {"original_key": "observation.images.front"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    (out / "meta" / "modality.json").write_text(json.dumps(modality, indent=4))
    print(f"done: {out} (episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}) + modality.json")


if __name__ == "__main__":
    main()
