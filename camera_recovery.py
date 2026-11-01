"""Experimental blackout detection and stop/resume wrapper for simulated cameras.

This only detects missing/invalid/all-black images. It does not establish real
robot safety or detect every camera failure (for example a frozen valid frame).
"""

from __future__ import annotations

import numpy as np

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


class CameraRecoveryPolicy:
    """Hold the last gripper command during blackout; replan on recovery."""

    def __init__(self, policy, required_valid_frames=3):
        self.policy = policy
        self.required_valid_frames = required_valid_frames
        self.reset()

    def reset(self):
        self.paused = False
        self.valid_frames = 0
        self.last_gripper = -1.0
        self.step = 0
        self.events = []
        self.monitor_log = []

    def get_action(self, obs):
        invalid = []
        for key in CAMERAS:
            value = obs.get(key)
            if value is None:
                invalid.append(key)
            else:
                image = np.asarray(value)
                if image.shape != (128, 128, 3) or not np.isfinite(image).all() or not np.any(image):
                    invalid.append(key)
        if invalid:
            if not self.paused:
                self.events.append({"step": self.step, "event": "pause", "invalid_cameras": invalid})
            self.paused = True
            self.valid_frames = 0
        elif self.paused:
            self.valid_frames += 1
            if self.valid_frames >= self.required_valid_frames:
                # Discard plans predicted before the blackout. Do not reseed the
                # model; predict a fresh chunk from the current observation.
                self.policy.scheduler.reset()
                self.paused = False
                self.events.append({"step": self.step, "event": "resume_with_new_plan"})
        if self.paused:
            action = np.zeros(7, dtype=np.float32)
            action[6] = self.last_gripper
        else:
            action = self.policy.get_action(obs)
            self.last_gripper = float(action[6])
        self.monitor_log.append({"step": self.step, "paused": self.paused, "invalid_cameras": invalid})
        self.step += 1
        return action
