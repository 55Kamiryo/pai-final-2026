"""Exploratory one-initial-state camera experiment with optional input recording."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def clear_device_cache(torch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/camera-recovery")
    parser.add_argument("--capture-inputs", action="store_true")
    parser.add_argument("--include-normal", action="store_true", help="Also record a no-fault reference rollout")
    args = parser.parse_args()

    import run_demo
    original_argv = sys.argv[:]
    sys.argv = ["run_demo.py", "--output", str(args.output_root / "bootstrap")]
    run_demo.configure(run_demo.arguments())
    import offline_policy
    import torch
    from camera_recovery import CameraRecoveryPolicy

    original_action = offline_policy.SmolVLALiberoPolicy.get_action
    original_prediction = offline_policy.SmolVLALiberoPolicy._predict_chunk
    try:
        for mode in (["normal"] if args.include_normal else []) + ["unprotected", "guarded"]:
            step = 0
            guard = None
            wrist_images = []
            prediction_steps = []

            def predict(self, obs):
                prediction_steps.append(step)
                return original_prediction(self, obs)

            def injected(self, obs):
                nonlocal step, guard
                modified = dict(obs)
                if mode != "normal" and 20 <= step < 60:
                    modified["robot0_eye_in_hand_image"] = np.zeros_like(obs["robot0_eye_in_hand_image"])
                if args.capture_inputs:
                    # Same orientation change as policy._image(); record actual
                    # pixels after fault injection, before executing the action.
                    wrist_images.append(np.ascontiguousarray(modified["robot0_eye_in_hand_image"][::-1, ::-1]))
                if mode == "guarded":
                    if guard is None:
                        adapter = SimpleNamespace(scheduler=self.scheduler, get_action=lambda value: original_action(self, value))
                        guard = CameraRecoveryPolicy(adapter)
                    action = guard.get_action(modified)
                else:
                    action = original_action(self, modified)
                step += 1
                return action

            offline_policy.SmolVLALiberoPolicy.get_action = injected
            offline_policy.SmolVLALiberoPolicy._predict_chunk = predict
            dest = args.output_root / mode
            sys.argv = ["run_demo.py", "--output", str(dest), "--episodes", "1", "--variants", "plain"]
            print("RECOVERY_CONDITION", mode, flush=True)
            run_demo.main()
            result_path = dest / "results.json"
            data = json.loads(result_path.read_text())
            data["camera_fault"] = None if mode == "normal" else {
                "camera": "robot0_eye_in_hand_image", "start_step": 20, "end_step_exclusive": 60,
                "type": "injected all-black image; camera restored after 2 simulated seconds",
            }
            data["camera_recovery_enabled"] = mode == "guarded"
            data["prediction_steps"] = prediction_steps
            if guard:
                data["camera_recovery_events"] = guard.events
                data["paused_steps"] = sum(e["paused"] for e in guard.monitor_log)
                (dest / "monitor.json").write_text(json.dumps(guard.monitor_log, indent=2) + "\n")
                print("GUARD_EVENTS", json.dumps(guard.events), flush=True)
            if args.capture_inputs:
                images = np.stack(wrist_images)
                n = len(images)
                predicted = np.zeros(n, dtype=bool)
                predicted[prediction_steps] = True
                paused = np.array([e["paused"] for e in guard.monitor_log], dtype=bool) if guard else np.zeros(n, dtype=bool)
                np.savez_compressed(
                    dest / "wrist_inputs.npz", images=images, step=np.arange(n),
                    simulation_seconds=np.arange(n) / 20, model_prediction=predicted,
                    paused=paused, injected_blackout=(np.arange(n) >= 20) & (np.arange(n) < 60) & (mode != "normal"),
                )
                data["input_capture"] = {
                    "file": "wrist_inputs.npz", "frames": n, "fps": 20,
                    "pixels": "actual post-fault-injection wrist RGB pixels, rotated 180 degrees exactly as policy._image()",
                    "alignment": "input step i is paired with external-camera frame i, before action i",
                    "model_use": "camera input recorded each control step; model uses the current image only at prediction_steps; guard consumes no model predictions while paused",
                }
            result_path.write_text(json.dumps(data, indent=2) + "\n")
            gc.collect()
            clear_device_cache(torch)
    finally:
        offline_policy.SmolVLALiberoPolicy.get_action = original_action
        offline_policy.SmolVLALiberoPolicy._predict_chunk = original_prediction
        sys.argv = original_argv


if __name__ == "__main__":
    main()
