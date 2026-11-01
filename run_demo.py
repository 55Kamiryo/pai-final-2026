"""Record real, closed-loop SmolVLA rollouts and compare action smoothing.

No training, network, telemetry, or cloud credentials are used during a run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parc-root", type=Path, help="Reuse existing local assets")
    p.add_argument("--asset-root", type=Path, default=Path("assets"))
    p.add_argument("--output", type=Path, default=Path("outputs/latest"))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--variants", nargs="+", choices=["plain", "smooth"], default=["plain", "smooth"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def configure(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.parc_root:
        root = args.parc_root.resolve()
        libero = root / ".deps/LIBERO-plus"
        model = root / "outputs/experiments/smolvla_libero_spatial_lora/smoke/base_model"
        config = model
        vlm = root / f".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/{VLM_REVISION}"
    else:
        root = args.asset_root.resolve()
        libero, model, config, vlm = root / "LIBERO-plus", root / "model", root / "model", root / "vlm"
    for path in [libero, model, config, vlm]:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing asset directory: {path}")
    package = libero / "libero/libero"
    # Keep LIBERO's config isolated from the user's ~/.libero/config.yaml.
    cfg = out / ".libero"
    cfg.mkdir(exist_ok=True)
    import yaml
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"), "assets": str(package / "assets"),
        "datasets": str(libero / "libero/datasets"),
    }))
    os.environ.update({
        "LIBERO_CONFIG_PATH": str(cfg), "MUJOCO_GL": "glfw" if sys.platform == "darwin" else "egl",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
    })
    sys.path.insert(0, str(libero))
    return out, package, model, config, vlm


def main():
    args = arguments()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max-steps must be positive")
    out, package, model, config, vlm = configure(args)
    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from offline_policy import InferenceSettings, SmolVLALiberoPolicy

    settings = {
        "plain": InferenceSettings(action_steps=25),
        "smooth": InferenceSettings(action_steps=25, blend_steps=5, ema_alpha=0.6),
    }
    policy = SmolVLALiberoPolicy(smolvla_root=model, config_root=config, vlm_assets_root=vlm)
    bddl = package / "bddl_files/libero_spatial" / f"{TASK}.bddl"
    initial_states = torch.load(package / "init_files/libero_spatial" / f"{TASK}.pruned_init", weights_only=False)
    if args.episodes > len(initial_states):
        raise ValueError("Not enough distinct saved initial states")
    summary = {
        "task": TASK, "instruction": TASK.replace("_", " "),
        "model": "lerobot/smolvla_libero_plus", "model_revision": "7bb70aa5bc92b82c9239142775d3a173103567ff",
        "model_sha256": hashlib.file_digest((model / "model.safetensors").open("rb"), "sha256").hexdigest(),
        "training_in_this_project": False, "control_hz": 20, "max_steps": args.max_steps,
        "camera_observation_size": 128, "video_size": 512, "video_fps": 20,
        "success_definition": "LIBERO task goal; collision-free competition scoring is not applied",
        "action_change_definition": "Mean L2 norm of consecutive differences in the six normalized arm action channels; excludes gripper. Not physical jerk.",
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "lerobot", "mujoco", "robosuite", "numpy", "transformers"]},
        "device": str(policy.device), "settings": {v: vars(settings[v]) for v in args.variants},
        "episodes": [],
    }
    for variant in args.variants:
        policy.settings = settings[variant]
        from offline_policy import ChunkScheduler
        policy.scheduler = ChunkScheduler(settings[variant], chunk_size=50)
        for ep in range(args.episodes):
            seed = args.seed + ep
            print(f"START {variant} initial_state={ep} seed={seed}", flush=True)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128, control_freq=20)
            stem = f"{variant}_{ep:02d}"
            video = out / f"{stem}.mp4"
            encoder = None
            try:
                env.seed(seed)
                env.reset()
                obs = env.set_init_state(initial_states[ep])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                if env.check_success():
                    raise RuntimeError("Initial state already satisfies the goal")
                policy.reset(summary["instruction"], seed=seed)
                encoder = subprocess.Popen([
                    "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "512x512", "-r", "20", "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(video),
                ], stdin=subprocess.PIPE)

                def record():
                    frame = env.sim.render(camera_name="agentview", width=512, height=512)
                    # Match LIBERO/LeRobot's displayed camera orientation.
                    encoder.stdin.write(np.ascontiguousarray(frame[::-1, ::-1]).tobytes())

                actions, positions, latencies = [], [obs["robot0_eef_pos"].copy()], []
                record()
                started = time.perf_counter()
                success = False
                for step in range(args.max_steps):
                    t = time.perf_counter()
                    action = policy.get_action(obs)
                    latencies.append(time.perf_counter() - t)
                    # The controller's valid action range is [-1, 1].
                    action = np.clip(action, -1.0, 1.0)
                    obs, reward, done, info = env.step(action)
                    actions.append(action.copy())
                    positions.append(obs["robot0_eef_pos"].copy())
                    record()
                    success = bool(env.check_success())
                    if success:
                        break
                elapsed = time.perf_counter() - started
                a, xyz = np.asarray(actions), np.asarray(positions)
                np.savez_compressed(out / f"{stem}.npz", actions=a, ee_positions=xyz, policy_call_seconds=latencies)
                row = {
                    "variant": variant, "initial_state_index": ep, "seed": seed,
                    "success": success, "steps": len(a), "simulated_seconds": len(a) / 20,
                    "wall_seconds": elapsed, "path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                    "mean_arm_action_change": float(np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1).mean()) if len(a)>1 else 0.0,
                    "video": video.name, "trajectory": f"{stem}.npz",
                }
                summary["episodes"].append(row)
                print("RESULT " + json.dumps(row), flush=True)
            finally:
                if encoder is not None:
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError("Video encoding failed")
                env.close()
            (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {len(summary['episodes'])} episodes", flush=True)


if __name__ == "__main__":
    main()
