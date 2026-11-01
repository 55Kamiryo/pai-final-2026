"""Build a synchronized recovery comparison with the recorded wrist input inset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FPS = 20
BG, WHITE, MUTED = "#101D2A", "#F2F6F8", "#A8BAC8"
TEAL, BLUE, RED = "#59D4BB", "#8BB9F6", "#FF8996"


def decode(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 512, 512, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/camera-recovery-inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "output/camera-recovery-preview-with-inputs.mp4")
    parser.add_argument("--font", type=Path, default=Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"))
    args = parser.parse_args()
    fonts = {n: ImageFont.truetype(str(args.font), n) for n in [16, 18, 20, 22, 26, 30]}
    cases = {}
    for name in ["unprotected", "guarded"]:
        folder = args.input / name
        data = json.loads((folder / "results.json").read_text())
        episode = data["episodes"][0]
        with np.load(folder / data["input_capture"]["file"], allow_pickle=False) as capture:
            inputs = {key: capture[key] for key in capture.files}
        frames = decode(folder / episode["video"])
        n = episode["steps"]
        assert inputs["images"].shape == (n, 128, 128, 3)
        assert len(frames) == n + 1
        assert np.array_equal(inputs["step"], np.arange(n))
        assert np.array_equal(inputs["simulation_seconds"], np.arange(n) / FPS)
        assert np.all(inputs["images"][inputs["injected_blackout"]] == 0)
        assert np.all(inputs["images"][~inputs["injected_blackout"]].any(axis=(1, 2, 3)))
        cases[name] = {"data": data, "episode": episode, "inputs": inputs, "frames": frames}

    fault = cases['guarded']['data']['camera_fault']
    if fault != cases['unprotected']['data']['camera_fault']:
        raise ValueError('The camera fault conditions must match')
    fault_start = fault['start_step']/FPS
    fault_end = fault['end_step_exclusive']/FPS
    resume_step = next(e['step'] for e in cases['guarded']['data']['camera_recovery_events'] if e['event'] == 'resume_with_new_plan')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = args.input / "preview-with-inputs"
    preview_dir.mkdir(exist_ok=True)
    total_frames = max(len(c["frames"]) for c in cases.values()) + FPS
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ], stdin=subprocess.PIPE)

    def text(draw, xy, value, size=20, color=WHITE):
        draw.text(xy, value, font=fonts[size], fill=color)

    try:
        for index in range(total_frames):
            image = Image.new("RGB", (1280, 720), BG)
            draw = ImageDraw.Draw(image)
            text(draw, (48, 16), f"作業中に手元カメラの入力を{fault_end-fault_start:g}秒間黒くする", 30)
            text(draw, (48, 62), f"同じ初期配置・同じAI / 手元画像だけを人工的に黒画面へ置換（{fault_start:g}〜{fault_end:g}秒）", 22, MUTED)
            for x, name, title in [(64, "unprotected", "対策なし"), (704, "guarded", "異常で停止・復旧後に再開")]:
                case = cases[name]
                frames, inputs, episode = case["frames"], case["inputs"], case["episode"]
                input_index = min(index, len(inputs["images"]) - 1)
                finished = index >= episode["steps"]
                fault = bool(inputs["injected_blackout"][input_index])
                paused = bool(inputs["paused"][input_index])
                text(draw, (x, 99), title, 26, BLUE if name == "unprotected" else TEAL)
                image.paste(Image.fromarray(frames[min(index, len(frames) - 1)]), (x, 140))
                draw.rectangle((x, 140, x + 512, 171), fill="#203B4A")
                text(draw, (x + 10, 144), "ロボット全体の映像（前方）", 18)

                # Actual control-input pixels saved during this same rollout.
                # The inset is before-action frame i, paired with scene frame i.
                inset_x, inset_y = x + 306, 213
                border = RED if fault else TEAL
                draw.rectangle((inset_x - 4, inset_y - 37, inset_x + 196, inset_y + 224), fill=BG, outline=border, width=2)
                text(draw, (inset_x + 3, inset_y - 29), "AI入力用の手元画像", 18)
                wrist = Image.fromarray(inputs["images"][input_index]).resize((192, 192), Image.Resampling.NEAREST)
                image.paste(wrist, (inset_x, inset_y))
                if finished:
                    input_status = "最終入力を静止表示"
                else:
                    input_status = "画像欠損：黒画面" if fault else "画像あり"
                text(draw, (inset_x + 4, inset_y + 198), input_status, 16, border)

                if finished:
                    status = "完了・最終状態を表示" if episode["success"] else "時間上限・未達成"
                elif name == "guarded" and paused:
                    status = "画像欠損 → 腕の移動指令を停止" if fault else "画像復旧 → 正常受信を確認中"
                elif name == "guarded" and index >= resume_step:
                    status = "現在の画像から計画を作り直して再開"
                else:
                    status = "AIが予測した動作を実行中"
                draw.rectangle((x, 612, x + 512, 652), fill=BG)
                text(draw, (x + 6, 619), status, 20, TEAL if name == "guarded" else WHITE)

            text(draw, (64, 658), f"{index / FPS:05.2f} 秒 / 前方画像は正常 / シミュレーション時間で等速（推論待ちを除く）", 20, MUTED)
            text(draw, (64, 692), "小窓は制御へ渡した画像。AIは動作列の更新時に参照。 / 本実証：1配置・各条件1回", 18, MUTED)
            if index in [10, 20, 40, 60, 62, 80, 140, 290]:
                image.save(preview_dir / f"frame_{index:03d}.png")
            encoder.stdin.write(image.tobytes())
    finally:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    (preview_dir / "build_info.json").write_text(json.dumps({
        "video": args.output.name, "frames": total_frames, "fps": FPS,
        "duration_seconds": total_frames / FPS,
        "inset_source": "wrist_inputs.npz: actual post-injection, pre-action pixels; not reconstructed footage",
        "alignment": "frame i shows observation before action i; after completion, final input is held and labelled",
    }, indent=2) + "\n")
    print(f"VIDEO {args.output} ({total_frames / FPS:.2f}s)")


if __name__ == "__main__":
    main()
