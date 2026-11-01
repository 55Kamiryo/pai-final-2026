"""Fetch public, pinned upstream dependencies; never bundle them in source ZIP."""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def adapt_model_config(config):
    """Correct the published legacy camera/state interface for LIBERO inference.

    The checkpoint's normalizer describes an 8D state and front/wrist images.
    This changes interface metadata only; model tensor weights are untouched.
    """
    config.update({
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.state": {"type": "STATE", "shape": [8]},
        },
        "empty_cameras": 0, "load_vlm_weights": False,
        "device": None, "repo_id": None, "pretrained_path": None, "push_to_hub": False,
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    args = parser.parse_args()
    root = args.asset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    libero = root / "LIBERO-plus"
    if not (libero / ".git").exists():
        subprocess.run(["git", "clone", "https://github.com/sylvestf/LIBERO-plus.git", str(libero)], check=True)
    subprocess.run(["git", "-C", str(libero), "checkout", "--detach", "4976dc30028e805ff8094b55501d532c48fec182"], check=True)
    assets = libero / "libero/libero/assets"
    if not assets.is_dir():
        archive = hf_hub_download(repo_id="Sylvest/LIBERO-plus", repo_type="dataset", filename="assets.zip", revision="dd2bd61b7d9a6fef1abc52d606e983b41886a149", token=False)
        with tempfile.TemporaryDirectory() as temp:
            with ZipFile(archive) as z:
                z.extractall(temp)
            candidates = sorted((p for p in Path(temp).rglob("assets") if p.is_dir()), key=lambda p: len(p.parts))
            if not candidates:
                raise FileNotFoundError("assets directory not present in upstream archive")
            shutil.move(str(candidates[0]), str(assets))
    snapshot_download(
        repo_id="lerobot/smolvla_libero_plus", revision=MODEL_REVISION, local_dir=root / "model", token=False,
        allow_patterns=["config.json", "model.safetensors", "policy_preprocessor*", "policy_postprocessor*"],
    )
    snapshot_download(
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", revision=VLM_REVISION,
        local_dir=root / "vlm", token=False,
        allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"],
    )
    # Model checkpoint already contains the VLM weights. Do not request them online
    # again when constructing the policy at inference time.
    config_path = root / "model/config.json"
    config = json.loads(config_path.read_text())
    config = adapt_model_config(config)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("Assets ready. Run: uv run python run_demo.py")


if __name__ == "__main__":
    main()
