from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from turbovla.models import TurboVLAConfig, build_turbovla
from turbovla.models.configuration import (
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    VisionEncoderConfig,
)


@dataclass
class TurboVLADefaultConfig:
    name: str = "TurboVLA"
    text: dict = field(
        default_factory=lambda: {
            "bert_path": "/path/to/bert-base-uncased",
            "max_text_len": 256,
            "sub_sentence_present": True,
            "local_files_only": True,
            "freeze_text_encoder": True,
            "attn_implementation": "flash_attention_2",
        }
    )
    vision: dict = field(
        default_factory=lambda: {
            "model_path": "/path/to/dinov3",
            "image_size": 224,
            "num_views": 3,
            "local_files_only": True,
            "freeze_vision_encoder": False,
            "attn_implementation": "flash_attention_2",
            "position_init_std": 0.01,
            "position_scale_init": 0.01,
            "dropout": 0.1,
        }
    )
    interaction: dict = field(
        default_factory=lambda: {
            "hidden_dim": 256,
            "nheads": 8,
            "dim_feedforward": 2048,
            "enhancer_inner_dim": 1024,
            "num_layers": 6,
            "text_dropout": 0.0,
            "fusion_dropout": 0.0,
            "fusion_droppath": 0.1,
            "padding_strategy": "zero_fill",
            "residual_style": "pre_norm",
            "attention_backend": "sdpa",
            "compute_precision": "bf16_autocast",
        }
    )
    initialization: dict = field(
        default_factory=lambda: {
            "pretrained_ckpt": "",
            "load_pretrained": True,
            "load_bert": True,
            "load_text_projection": True,
            "load_interaction": True,
        }
    )
    action: dict = field(
        default_factory=lambda: {
            "action_dim": 14,
            "state_dim": 14,
            "horizon": 50,
            "num_layers": 3,
            "num_state_tokens": 2,
            "state_hidden_dim": 256,
            "mlp_hidden_dim": 512,
            "dropout": 0.1,
            "loss_type": "l1",
        }
    )


@FRAMEWORK_REGISTRY.register("TurboVLA")
class TurboVLAFramework(baseframework):
    """StarVLA batch and checkpoint adapter around the shared TurboVLA model."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        del kwargs
        super().__init__()
        self.config = merge_framework_config(TurboVLADefaultConfig, config)
        fw = self.config.framework
        self._historical_all50 = self._align_historical_all50_config(fw)
        self.model = build_turbovla(self._core_config(fw))
        self.image_processor = AutoImageProcessor.from_pretrained(
            fw.vision.model_path,
            local_files_only=fw.vision.local_files_only,
        )
        self.image_size = int(fw.vision.image_size)
        self.num_views = int(fw.vision.num_views)
        if hasattr(self.image_processor, "size"):
            self.image_processor.size = {"height": self.image_size, "width": self.image_size}
        self.action_horizon = int(fw.action.horizon)
        self.loss_type = str(fw.action.loss_type).lower()
        if fw.initialization.load_pretrained:
            self._load_initialization(fw.initialization)

    @staticmethod
    def _align_historical_all50_config(fw) -> bool:
        """Interpret the 200k all50 checkpoint schema using the shared TurboVLA model."""
        if fw.get("dinov3") is None:
            return False

        dino = fw.dinov3
        fusion = fw.fusion
        action = fw.action_model
        groundingdino = fw.groundingdino
        if str(fw.text.get("encoder_type", "bert")).lower() != "bert":
            raise ValueError("Only BERT checkpoints are supported by this RoboTwin path")
        if str(action.get("action_model_type", "act")).lower() != "act":
            raise ValueError("Only ACT checkpoints are supported by this RoboTwin path")
        if int(action.get("act_hidden_dim", fusion.hidden_dim)) != int(fusion.hidden_dim):
            raise ValueError("ACT and fusion hidden dimensions must match")
        if int(action.get("act_nheads", fusion.nheads)) != int(fusion.nheads):
            raise ValueError("ACT and fusion attention-head counts must match")

        fw.text.bert_path = fw.text.get("model_path") or fw.text.bert_path
        fw.vision.model_path = dino.model_path
        fw.vision.image_size = dino.image_size
        fw.vision.num_views = dino.num_views
        fw.vision.local_files_only = dino.local_files_only
        fw.vision.freeze_vision_encoder = dino.freeze_vision_encoder
        fw.vision.attn_implementation = dino.get("attn_implementation")
        fw.vision.position_init_std = dino.vision_pos_init_std
        fw.vision.position_scale_init = dino.vision_pos_scale_init
        fw.vision.dropout = dino.vision_dropout

        fw.interaction.hidden_dim = fusion.hidden_dim
        fw.interaction.nheads = fusion.nheads
        fw.interaction.dim_feedforward = action.act_dim_feedforward
        fw.interaction.enhancer_inner_dim = fusion.enhancer_inner_dim
        fw.interaction.num_layers = fusion.num_layers
        fw.interaction.text_dropout = fusion.text_dropout
        fw.interaction.fusion_dropout = fusion.fusion_dropout
        fw.interaction.fusion_droppath = fusion.fusion_droppath
        fw.interaction.padding_strategy = "zero_fill"
        fw.interaction.residual_style = "pre_norm"
        fw.interaction.attention_backend = "sdpa"
        fw.interaction.compute_precision = "bf16_autocast"

        fw.action.action_dim = action.action_dim
        fw.action.state_dim = action.state_dim
        fw.action.horizon = action.action_horizon
        fw.action.num_layers = action.act_num_layers
        fw.action.num_state_tokens = action.act_state_tokens
        fw.action.state_hidden_dim = action.act_state_hidden_dim
        fw.action.mlp_hidden_dim = action.act_mlp_hidden_dim
        fw.action.dropout = action.act_dropout
        fw.action.loss_type = action.act_loss_type

        fw.initialization.pretrained_ckpt = groundingdino.pretrained_ckpt
        fw.initialization.load_pretrained = groundingdino.load_pretrained
        fw.initialization.load_bert = groundingdino.load_bert
        fw.initialization.load_text_projection = groundingdino.load_text_proj
        fw.initialization.load_interaction = groundingdino.load_feature_enhancer
        return True

    @staticmethod
    def _core_config(fw) -> TurboVLAConfig:
        return TurboVLAConfig(
            compatibility_profile="robotwin",
            text=TextEncoderConfig(
                model_name_or_path=fw.text.bert_path,
                max_length=int(fw.text.max_text_len),
                padding_length=None,
                sub_sentence_present=bool(fw.text.sub_sentence_present),
                frozen=bool(fw.text.freeze_text_encoder),
                force_eval_when_frozen=True,
                zero_padded_tokens=True,
                local_files_only=bool(fw.text.local_files_only),
                attention_implementation=fw.text.get("attn_implementation"),
            ),
            vision=VisionEncoderConfig(
                model_name_or_path=fw.vision.model_path,
                image_size=int(fw.vision.image_size),
                num_views=int(fw.vision.num_views),
                position_embedding="learned_patch",
                encode_views_separately=False,
                frozen=bool(fw.vision.freeze_vision_encoder),
                local_files_only=bool(fw.vision.local_files_only),
                attention_implementation=fw.vision.get("attn_implementation"),
                compute_precision="bf16_autocast",
                position_init_std=float(fw.vision.position_init_std),
                position_scale_init=float(fw.vision.position_scale_init),
                dropout=float(fw.vision.dropout),
            ),
            interaction=InteractionConfig(
                hidden_dim=int(fw.interaction.hidden_dim),
                nheads=int(fw.interaction.nheads),
                num_layers=int(fw.interaction.num_layers),
                dim_feedforward=int(fw.interaction.dim_feedforward),
                enhancer_inner_dim=int(fw.interaction.enhancer_inner_dim),
                text_dropout=float(fw.interaction.text_dropout),
                fusion_dropout=float(fw.interaction.fusion_dropout),
                fusion_droppath=float(fw.interaction.fusion_droppath),
                padding_strategy=str(fw.interaction.padding_strategy),
                residual_style=str(fw.interaction.residual_style),
                attention_backend=str(fw.interaction.attention_backend),
                compute_precision=str(fw.interaction.compute_precision),
            ),
            action=ActionHeadConfig(
                action_dim=int(fw.action.action_dim),
                state_dim=int(fw.action.state_dim),
                horizon=int(fw.action.horizon),
                num_state_tokens=int(fw.action.num_state_tokens),
                num_layers=int(fw.action.num_layers),
                mlp_hidden_dim=int(fw.action.mlp_hidden_dim),
                state_hidden_dim=int(fw.action.state_hidden_dim),
                dropout=float(fw.action.dropout),
            ),
        )

    @staticmethod
    def _checkpoint_state(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if isinstance(checkpoint.get(key), dict):
                    return checkpoint[key]
        return checkpoint

    @staticmethod
    def _historical_all50_key(key: str) -> str:
        """Map an all50 checkpoint tensor to the shared TurboVLA module layout."""
        exact = {
            "view_embed": "model.view_embedding",
            "vision_pos_embed": "model.patch_position_embedding",
            "vision_pos_scale": "model.patch_position_scale",
            "action_model.state_proj.pos": "model.action_head.state_projection.position",
        }
        if key in exact:
            return exact[key]
        prefixes = (
            ("text_encoder.bert.", "model.text_encoder.bert."),
            ("text_encoder.text_proj.", "model.text_encoder.text_projection."),
            ("dinov3.model.", "model.vision_encoder.backbone."),
            ("vision_proj.norm_in.", "model.vision_projection.input_norm."),
            ("vision_proj.norm_out.", "model.vision_projection.output_norm."),
            ("vision_proj.mlp.", "model.vision_projection.mlp."),
            ("vision_proj.skip.", "model.vision_projection.skip."),
            ("feature_enhancer.", "model.vision_language_interaction."),
            ("action_model.state_proj.net.", "model.action_head.state_projection.net."),
            ("action_model.state_proj.out_norm.", "model.action_head.state_projection.output_norm."),
            ("action_model.action_policy.action_queries.", "model.action_head.decoder.action_queries."),
            ("action_model.action_policy.decoder.", "model.action_head.decoder.decoder."),
            ("action_model.action_policy.action_head.layers.", "model.action_head.decoder.action_projection.layers."),
        )
        for source, target in prefixes:
            if key.startswith(source):
                return target + key[len(source):]
        raise KeyError(f"Unrecognized historical all50 checkpoint tensor: {key}")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self._historical_all50 and state_dict and not any(
            key.startswith("model.") for key in state_dict
        ):
            translated = {}
            for key, value in state_dict.items():
                target = self._historical_all50_key(key)
                if target in translated:
                    raise ValueError(f"Duplicate mapped checkpoint tensor: {target}")
                translated[target] = value
            state_dict = translated
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_initialization(self, init_cfg) -> None:
        path = str(init_cfg.pretrained_ckpt)
        if not path:
            raise ValueError("framework.initialization.pretrained_ckpt is required")
        source = self._checkpoint_state(torch.load(path, map_location="cpu"))
        source = {(key[7:] if key.startswith("module.") else key): value for key, value in source.items()}
        mappings = []
        if init_cfg.load_bert:
            mappings.append(("bert.", "text_encoder.bert."))
        if init_cfg.load_text_projection:
            mappings.append(("feat_map.", "text_encoder.text_projection."))
        if init_cfg.load_interaction:
            mappings.extend(
                [
                    ("transformer.encoder.text_layers.", "vision_language_interaction.text_layers."),
                    ("transformer.encoder.fusion_layers.", "vision_language_interaction.fusion_layers."),
                ]
            )
        target = self.model.state_dict()
        loaded = {}
        for source_key, value in source.items():
            for source_prefix, target_prefix in mappings:
                if not source_key.startswith(source_prefix):
                    continue
                target_key = target_prefix + source_key[len(source_prefix) :]
                if target_key in target and tuple(target[target_key].shape) == tuple(value.shape):
                    loaded[target_key] = value
                break
        if not loaded:
            raise RuntimeError(f"no compatible initialization tensors found in {path}")
        target.update(loaded)
        self.model.load_state_dict(target, strict=True)
        print(f"[TurboVLA] loaded {len(loaded)} initialization tensors from {path}", flush=True)

    @staticmethod
    def _as_view_list(images, num_views: int) -> list[Image.Image]:
        images = to_pil_preserve(images)
        views = [images] if isinstance(images, Image.Image) else list(images)
        if not views:
            raise ValueError("each example must contain at least one image")
        if len(views) < num_views:
            views.extend([views[-1]] * (num_views - len(views)))
        return views[:num_views]

    def _model_inputs(self, examples: List[dict]):
        if not isinstance(examples, list):
            examples = [examples]
        device = next(self.parameters()).device
        views = [self._as_view_list(example["image"], self.num_views) for example in examples]
        flat_images = [image for example_views in views for image in example_views]
        pixel_values = self.image_processor(images=flat_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.view(len(examples), self.num_views, *pixel_values.shape[1:]).to(device)
        instructions = [str(example["lang"]) for example in examples]
        states = torch.as_tensor(
            np.asarray([example["state"] for example in examples]),
            device=device,
            dtype=torch.float32,
        )
        return instructions, {"dinov3": pixel_values}, states

    def forward(self, examples: List[dict] = None, **kwargs):
        del kwargs
        instructions, samples, states = self._model_inputs(examples)
        predicted = self.model(instructions, samples, states)
        targets = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=predicted.device,
            dtype=predicted.dtype,
        )[:, -self.action_horizon :]
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, targets)
        elif self.loss_type in {"smooth_l1", "huber"}:
            loss = F.smooth_l1_loss(predicted, targets)
        else:
            loss = F.l1_loss(predicted, targets)
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        profile_latency = bool(kwargs.pop("profile_latency", False))
        start = time.perf_counter()
        instructions, samples, states = self._model_inputs(examples)
        predicted = self.model(instructions, samples, states)
        output = {"normalized_actions": predicted.detach().float().cpu().numpy()}
        if profile_latency:
            if predicted.device.type == "cuda":
                torch.cuda.synchronize(predicted.device)
            output["latency_ms"] = {"predict_action_total": (time.perf_counter() - start) * 1000.0}
        return output

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def text_encoder(self):
        return self.model.text_encoder

    @property
    def vision_encoder(self):
        return self.model.vision_encoder

    @property
    def vision_language_interaction(self):
        return self.model.vision_language_interaction

    @property
    def vision_projection(self):
        return self.model.vision_projection

    @property
    def action_head(self):
        return self.model.action_head
