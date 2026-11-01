"""Offline SmolVLA inference adapter for the PARC2026 policy interface."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


MODEL_ROOT = Path(__file__).resolve().parent / "model_weights"
SMOLVLA_ROOT = MODEL_ROOT / "smolvla"
VLM_ASSETS_ROOT = MODEL_ROOT / "vlm_assets"
INFERENCE_CONFIG_PATH = MODEL_ROOT / "inference_config.json"
REQUIRED_OBSERVATIONS = {
    "agentview_image": (128, 128, 3),
    "robot0_eye_in_hand_image": (128, 128, 3),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
}
GRIPPER_DIM = 6


@dataclass
class InferenceSettings:
    """Action-chunk scheduling knobs applied on top of the trained policy.

    All post-processing operates on the policy's own action chunks; the model
    remains the sole source of actions.
    """

    # How many actions of each 50-step chunk are executed before replanning.
    action_steps: int = 25
    # Crossfade length (env steps) between the previous chunk's remaining plan
    # and a freshly predicted chunk. 0 disables blending.
    blend_steps: int = 0
    # Exponential moving average factor on the delta-action output
    # (1.0 = no smoothing; the gripper channel is never smoothed).
    ema_alpha: float = 1.0
    # Predict a new chunk every `ensemble_stride` env steps and average all
    # live chunks per ACT-style temporal ensembling. 0 disables ensembling.
    ensemble_stride: int = 0
    # Exponential weight decay per chunk-age rank when ensembling
    # (0.0 = uniform average; larger favours older, more settled plans).
    ensemble_temperature: float = 0.35

    def __post_init__(self) -> None:
        if self.action_steps < 1:
            raise ValueError("action_steps must be >= 1")
        if self.blend_steps < 0:
            raise ValueError("blend_steps must be >= 0")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.ensemble_stride < 0:
            raise ValueError("ensemble_stride must be >= 0")

    @classmethod
    def load(cls, path: Path, **overrides) -> "InferenceSettings":
        data: dict = {}
        if path.is_file():
            data = json.loads(path.read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Match LeRobot's LIBERO (x, y, z, w) conversion exactly."""
    quaternion = quaternion.to(dtype=torch.float32)
    w = quaternion[:, 3].clamp(-1.0, 1.0)
    denominator = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    result = torch.zeros((quaternion.shape[0], 3), device=quaternion.device)
    mask = denominator > 1e-10
    if mask.any():
        angle = 2.0 * torch.acos(w[mask])
        axis = quaternion[mask, :3] / denominator[mask].unsqueeze(1)
        result[mask] = axis * angle.unsqueeze(1)
    return result


def _load_stats(path: Path, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = load_file(str(path), device="cpu")
    return tensors[f"{prefix}.mean"], tensors[f"{prefix}.std"]


class ChunkScheduler:
    """Schedule per-step actions out of full policy chunks.

    Three modes, all fed exclusively by `predict_chunk` (the trained policy):
    - plain: execute `action_steps` actions per chunk, then replan.
    - blend: same, but crossfade the first `blend_steps` actions of a new
      chunk with the previous chunk's remaining plan to remove boundary jerk.
    - ensemble: replan every `ensemble_stride` steps and average all chunks
      covering the current step (ACT-style temporal ensembling).
    The gripper channel always comes from the newest chunk unmixed, and the
    optional EMA low-pass never touches it either.
    """

    def __init__(self, settings: InferenceSettings, chunk_size: int) -> None:
        self.settings = settings
        self.chunk_size = int(chunk_size)
        if settings.action_steps > self.chunk_size:
            raise ValueError(
                f"action_steps must be <= chunk size {self.chunk_size}"
            )
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._plan: list[np.ndarray] = []
        self._prev_tail: np.ndarray | None = None
        self._chunks: list[tuple[int, np.ndarray]] = []
        self._last_output: np.ndarray | None = None

    def step(self, predict_chunk) -> np.ndarray:
        if self.settings.ensemble_stride > 0:
            action = self._ensemble_step(predict_chunk)
        else:
            action = self._plan_step(predict_chunk)
        action = self._ema(action)
        self._step += 1
        return action

    def _plan_step(self, predict_chunk) -> np.ndarray:
        if not self._plan:
            chunk = np.asarray(predict_chunk(), dtype=np.float32)
            executed = chunk[: self.settings.action_steps].copy()
            blend = min(self.settings.blend_steps, len(executed))
            if blend > 0 and self._prev_tail is not None:
                usable = min(blend, len(self._prev_tail))
                for i in range(usable):
                    w = (i + 1.0) / (usable + 1.0)
                    executed[i, :GRIPPER_DIM] = (
                        (1.0 - w) * self._prev_tail[i, :GRIPPER_DIM]
                        + w * executed[i, :GRIPPER_DIM]
                    )
            tail = chunk[
                self.settings.action_steps :
                self.settings.action_steps + max(self.settings.blend_steps, 1)
            ]
            self._prev_tail = tail.copy() if len(tail) else None
            self._plan = list(executed)
        return self._plan.pop(0)

    def _ensemble_step(self, predict_chunk) -> np.ndarray:
        stride = self.settings.ensemble_stride
        if not self._chunks or (self._step % stride == 0):
            chunk = np.asarray(predict_chunk(), dtype=np.float32)
            self._chunks.append((self._step, chunk))
        # Drop chunks that no longer cover the current step.
        self._chunks = [
            (start, chunk)
            for start, chunk in self._chunks
            if self._step - start < len(chunk)
        ]
        if not self._chunks:  # pragma: no cover - guarded by append above
            raise RuntimeError("No action chunk covers the current step")
        preds = []
        for start, chunk in self._chunks:
            preds.append(chunk[self._step - start])
        stacked = np.stack(preds, axis=0)
        ranks = np.arange(len(stacked), dtype=np.float32)  # 0 = oldest
        weights = np.exp(-self.settings.ensemble_temperature * ranks)
        weights /= weights.sum()
        action = (weights[:, None] * stacked).sum(axis=0).astype(np.float32)
        action[GRIPPER_DIM] = stacked[-1, GRIPPER_DIM]
        return action

    def _ema(self, action: np.ndarray) -> np.ndarray:
        alpha = self.settings.ema_alpha
        if alpha < 1.0 and self._last_output is not None:
            action = action.copy()
            action[:GRIPPER_DIM] = (
                alpha * action[:GRIPPER_DIM]
                + (1.0 - alpha) * self._last_output[:GRIPPER_DIM]
            )
        self._last_output = action
        return action


class SmolVLALiberoPolicy:
    """Adapt raw evaluator observations to a merged SmolVLA action policy."""

    def __init__(
        self,
        *,
        smolvla_root: Path | None = None,
        config_root: Path | None = None,
        vlm_assets_root: Path | None = None,
        action_steps: int | None = None,
        inference_seed: int = 2026,
        inference_settings: InferenceSettings | None = None,
    ) -> None:
        self.smolvla_root = Path(smolvla_root or SMOLVLA_ROOT).resolve()
        self.config_root = Path(config_root or self.smolvla_root).resolve()
        self.vlm_assets_root = Path(vlm_assets_root or VLM_ASSETS_ROOT).resolve()
        self.preprocessor_state = (
            self.smolvla_root
            / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        )
        self.postprocessor_state = (
            self.smolvla_root
            / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        missing = [
            path
            for path in (
                self.smolvla_root,
                self.config_root / "config.json",
                self.vlm_assets_root,
                self.preprocessor_state,
                self.postprocessor_state,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Submission model assets are missing: {missing}")

        # Fail rather than reaching the network if an asset was omitted from the ZIP.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.device = _device()
        self.instruction = ""
        self.inference_seed = int(inference_seed)

        # The vendored LeRobot 0.4.3 inference code supports the scoring
        # environment's Python 3.10.  The checkpoint itself is produced by 0.6.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(
            self.config_root,
            local_files_only=True,
        )
        config.device = self.device.type
        config.vlm_model_name = str(self.vlm_assets_root)
        if inference_settings is None:
            inference_settings = InferenceSettings.load(
                INFERENCE_CONFIG_PATH, action_steps=action_steps
            )
        elif action_steps is not None:
            raise ValueError(
                "Pass action_steps through inference_settings when both are given"
            )
        if not 1 <= inference_settings.action_steps <= int(config.chunk_size):
            raise ValueError(
                f"action_steps must be in [1, {config.chunk_size}], "
                f"got {inference_settings.action_steps}"
            )
        self.settings = inference_settings
        config.n_action_steps = inference_settings.action_steps
        self.policy = SmolVLAPolicy.from_pretrained(
            self.smolvla_root,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.policy.eval()
        self.scheduler = ChunkScheduler(
            self.settings, chunk_size=int(config.chunk_size)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.vlm_assets_root,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"

        state_mean, state_std = _load_stats(
            self.preprocessor_state, "observation.state"
        )
        action_mean, action_std = _load_stats(self.postprocessor_state, "action")
        self.state_mean = state_mean.to(self.device)
        self.state_std = state_std.to(self.device)
        self.action_mean = action_mean.to(self.device)
        self.action_std = action_std.to(self.device)

        config_json = json.loads((self.config_root / "config.json").read_text())
        self.tokenizer_max_length = int(config_json["tokenizer_max_length"])
        self.pad_language_to = config_json["pad_language_to"]
        self.reset("")

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.instruction = str(instruction)
        episode_seed = self.inference_seed if seed is None else int(seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        self.policy.reset()
        self.scheduler.reset()

    def _validate_observation(self, obs: dict[str, np.ndarray]) -> None:
        for key, expected_shape in REQUIRED_OBSERVATIONS.items():
            if key not in obs:
                raise KeyError(f"Required observation is missing: {key}")
            if tuple(np.asarray(obs[key]).shape) != expected_shape:
                raise ValueError(
                    f"Unexpected shape for {key}: {np.asarray(obs[key]).shape}; "
                    f"expected {expected_shape}"
                )

    def _image(self, value: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(value)).to(self.device)
        image = image.permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
        # LIBERO cameras use the opposite orientation from the training dataset.
        return torch.flip(image, dims=[1, 2]).unsqueeze(0)

    def _batch(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        self._validate_observation(obs)
        position = torch.as_tensor(
            obs["robot0_eef_pos"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        quaternion = torch.as_tensor(
            obs["robot0_eef_quat"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        gripper = torch.as_tensor(
            obs["robot0_gripper_qpos"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        state = torch.cat(
            (position, _quaternion_to_axis_angle(quaternion), gripper), dim=-1
        )
        state = (state - self.state_mean) / (self.state_std + 1e-8)

        task = self.instruction
        if not task.endswith("\n"):
            task += "\n"
        tokens = self.tokenizer(
            [task],
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding=self.pad_language_to,
            return_tensors="pt",
        )

        return {
            "observation.images.front": self._image(obs["agentview_image"]),
            "observation.images.wrist": self._image(
                obs["robot0_eye_in_hand_image"]
            ),
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].to(
                self.device, dtype=torch.bool
            ),
        }

    def _predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        normalized = self.policy.predict_action_chunk(self._batch(obs))
        chunk = normalized * self.action_std + self.action_mean
        chunk = chunk.squeeze(0).to(device="cpu", dtype=torch.float32).numpy()
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise RuntimeError(f"Unexpected action chunk shape: {chunk.shape}")
        return chunk

    @torch.inference_mode()
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        action = self.scheduler.step(lambda: self._predict_chunk(obs))
        action = np.asarray(action, dtype=np.float32).reshape(7)
        if not np.isfinite(action).all():
            raise RuntimeError("Policy produced a non-finite action")
        return action
