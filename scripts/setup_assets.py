"""
Pre-download the default model/data assets so the first `streamlit run` isn't
slow. Safe to re-run: each step is skipped if its target file already exists.

This script is intentionally standalone (it does not import pipeline.py) so it
can be run before the rest of the app is finished, and so it keeps working
even if pipeline.py's internal APIs change later.

Downloads:
  1. Frozen SimCLR ResNet18 encoder weights (VEDB-pretrained), from
     DM-Diaz/VEDB-SimCLR-ResNet18-Baseline on HuggingFace.
  2. Frozen ridge-regression voxel-encoding weights (subject S1, baseline
     model), from DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models on HuggingFace.
  3. The NSD V1/visual-ROI mask for subject 1 (prf-visualrois.nii.gz), fetched
     anonymously over HTTPS from the public NSD S3 bucket.
"""

import os
import sys

import requests
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets_cache")

ENCODER_REPO = "DM-Diaz/VEDB-SimCLR-ResNet18-Baseline"
RIDGE_REPO = "DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models"
RIDGE_FILENAME = "baseline/NSD_S1_resnet18-Baseline_concat.npy"
ROI_URL = (
    "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/"
    "subj01/func1pt8mm/roi/prf-visualrois.nii.gz"
)
ROI_FILENAME = "prf-visualrois_subj01.nii.gz"


def download_encoder_weights():
    """Find and download the frozen SimCLR ResNet18 encoder checkpoint."""
    print(f"[1/3] SimCLR encoder weights ({ENCODER_REPO})")
    files = list_repo_files(ENCODER_REPO)
    checkpoints = [f for f in files if f.endswith(".pth.tar")]
    if not checkpoints:
        print(f"  ERROR: no .pth.tar file found in {ENCODER_REPO}. Files seen: {files}")
        sys.exit(1)
    target = checkpoints[0]
    if len(checkpoints) > 1:
        print(f"  Note: multiple .pth.tar files found, using the first: {target}")

    local_path = os.path.join(ASSETS_DIR, os.path.basename(target))
    if os.path.exists(local_path):
        print(f"  Already present, skipping: {local_path}")
        return local_path

    print(f"  Downloading {target} ...")
    downloaded = hf_hub_download(repo_id=ENCODER_REPO, filename=target, local_dir=ASSETS_DIR)
    print(f"  Saved to {downloaded}")
    return downloaded


def download_ridge_weights():
    """Download the frozen NSD S1 ridge-regression encoding weights."""
    print(f"[2/3] Ridge-regression voxel encoding weights ({RIDGE_REPO})")
    local_path = os.path.join(ASSETS_DIR, os.path.basename(RIDGE_FILENAME))
    if os.path.exists(local_path):
        print(f"  Already present, skipping: {local_path}")
        return local_path

    print(f"  Downloading {RIDGE_FILENAME} ...")
    downloaded = hf_hub_download(repo_id=RIDGE_REPO, filename=RIDGE_FILENAME, local_dir=ASSETS_DIR)
    print(f"  Saved to {downloaded}")
    return downloaded


def download_roi_mask():
    """Download the public NSD V1/visual-ROI mask (anonymous HTTPS, no auth)."""
    print(f"[3/3] NSD visual-ROI mask ({ROI_URL})")
    local_path = os.path.join(ASSETS_DIR, ROI_FILENAME)
    if os.path.exists(local_path):
        print(f"  Already present, skipping: {local_path}")
        return local_path

    print("  Downloading prf-visualrois.nii.gz ...")
    response = requests.get(ROI_URL, timeout=60)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"  Saved to {local_path}")
    return local_path


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    print(f"Asset cache directory: {ASSETS_DIR}\n")

    download_encoder_weights()
    download_ridge_weights()
    download_roi_mask()

    print("\nAll default assets are ready.")
