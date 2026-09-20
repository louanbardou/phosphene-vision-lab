"""
Phosphene Vision Pipeline -- model/science logic.

A REAL (not mocked), zero-training pipeline:

    photo -> frozen CNN (SimCLR ResNet18) -> frozen NSD ridge-regression
    encoding weights (V1 voxels only) -> gradient-based inversion of a
    phosphene stimulation pattern -> real differentiable phosphene simulator
    (dynaphos)

Every weight used here is frozen and published by someone else. The only
thing this app ever optimizes is the stimulation-current vector fed to the
phosphene simulator. See `project_to_1201` below for the one deliberate,
disclosed approximation in the chain (a missing published PCA basis).

This module has no UI code. It is imported by app.py (built separately).
"""

from __future__ import annotations

import functools
import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from torchvision.models import resnet18

from dynaphos.cortex_models import (
    get_cortex_coordinates_grid,
    get_visual_field_coordinates_from_cortex_full,
)
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import load_params

# ==============================================================================
# Paths / constants
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
ASSETS_CACHE = REPO_ROOT / "assets_cache"
ASSETS_CACHE.mkdir(parents=True, exist_ok=True)
(ASSETS_CACHE / ".gitignore").write_text("*\n!.gitignore\n")

CONFIG_PATH = BASE_DIR / "config" / "params.yaml"

PROJECTION_CACHE_PATH = ASSETS_CACHE / "random_projection_seed0.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
MODEL_VARIANTS = [
    "VEDB Baseline",
    "VEDB Fovea-Gaze",
    "VEDB Periph",
    "VEDB Periph-NF",
    "ImageNet-1K (reference)",
    "ImageNet-100 (reference)",
]

BACKBONE_REPOS: Dict[str, str] = {
    "VEDB Baseline": "DM-Diaz/VEDB-SimCLR-ResNet18-Baseline",
    "VEDB Fovea-Gaze": "DM-Diaz/VEDB-SimCLR-ResNet18-Fovea-Gaze",
    "VEDB Periph": "DM-Diaz/VEDB-SimCLR-ResNet18-Periph",
    "VEDB Periph-NF": "DM-Diaz/VEDB-SimCLR-ResNet18-Periph-NF",
    "ImageNet-1K (reference)": "DM-Diaz/SimCLR-ResNet18-ImageNet1K",
    "ImageNet-100 (reference)": "DM-Diaz/SimCLR-ResNet18-ImageNet100",
}

NSD_REPO = "DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models"
# Directory prefixes inside the NSD encoding-models repo, confirmed via
# list_repo_files this session. Anchored with a trailing "/" when matched so
# e.g. "periph" never accidentally matches "periph-nf/...".
NSD_VARIANT_DIRS: Dict[str, str] = {
    "VEDB Baseline": "baseline",
    "VEDB Fovea-Gaze": "fovea-gaze",
    "VEDB Periph": "periph",
    "VEDB Periph-NF": "periph-nf",
    "ImageNet-1K (reference)": "reference-models/resnet18-simclr-imagenet1k",
    "ImageNet-100 (reference)": "reference-models/resnet18-simclr-imagenet100",
}

# The 6 activation points tapped from the frozen backbone, in the exact order
# the original authors concatenated them into their 1200-d feature vector
# (confirmed this session from the `features_file_list` field stored inside
# the NSD encoding-model .npy files: conv1, layer1-1, layer2-1, layer3-1,
# layer4-1, avgpool).
LAYER_KEYS = ["conv1", "layer1.1", "layer2.1", "layer3.1", "layer4.1", "avgpool"]

# Channel count of each tapped layer in a standard resnet18 -- fixed by the
# architecture, identical across every VEDB/ImageNet SimCLR variant, so a
# single cached random-projection file works for all of them.
PROJECTION_IN_DIMS: Dict[str, int] = {
    "conv1": 64,
    "layer1.1": 64,
    "layer2.1": 128,
    "layer3.1": 256,
    "layer4.1": 512,
    "avgpool": 512,
}
PROJECTION_OUT_DIM = 200


# ==============================================================================
# Component 1: frozen SimCLR ResNet18 backbone
# ==============================================================================
class SimCLRResNet18(nn.Module):
    """Matches the checkpoint's module tree exactly (backbone + 2-layer MLP
    projection head) so `load_state_dict(..., strict=True)` succeeds."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))


class FeatureBackbone(nn.Module):
    """Wraps a frozen resnet18 backbone (with its projection head already
    stripped to nn.Identity) and, on each forward pass, returns a dict of the
    6 intermediate activations tapped via forward hooks. Weights are frozen
    (requires_grad_(False)) but gradients still flow through the *input*
    tensor, which is what the phosphene-inversion loop needs.
    """

    def __init__(self, resnet_backbone: nn.Module):
        super().__init__()
        self.backbone = resnet_backbone
        self._features: Dict[str, torch.Tensor] = {}

        modules = dict(self.backbone.named_modules())
        missing = [k for k in LAYER_KEYS if k not in modules]
        if missing:
            raise KeyError(f"Backbone is missing expected modules: {missing}")

        for name in LAYER_KEYS:
            modules[name].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._features[name] = out

        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._features = {}
        self.backbone(x)  # fc is nn.Identity; hooks populate self._features
        return dict(self._features)


# The 2 "reference" repos (ImageNet-1K/ImageNet-100) ship PyTorch-Lightning
# checkpoints whose backbone is a plain nn.Sequential(*resnet18.children())
# rather than a torchvision resnet18 with named attributes, so their keys
# look like "backbone.0.weight" (numeric index) instead of
# "backbone.conv1.weight". Only the indices with learnable params appear;
# the rest (relu/maxpool/avgpool) are parameter-free and simply absent.
_LIGHTNING_SEQ_TO_RESNET_ATTR = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def _remap_lightning_backbone_state_dict(state_dict: dict) -> dict:
    """Convert a Lightning-style sequential-indexed backbone state dict (see
    `_LIGHTNING_SEQ_TO_RESNET_ATTR` above) into the attribute-named form a
    torchvision resnet18 expects, dropping any non-"backbone." keys (e.g.
    "projection_head.*") since the projection head is discarded regardless.
    """
    remapped = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue
        parts = k.split(".")
        seq_idx = parts[1]
        attr_name = _LIGHTNING_SEQ_TO_RESNET_ATTR.get(seq_idx)
        if attr_name is None:
            continue  # parameter-free child (relu/maxpool/avgpool) or unrecognized
        remapped[".".join([attr_name] + parts[2:])] = v
    return remapped


@functools.lru_cache(maxsize=None)
def load_backbone(model_variant: str) -> nn.Module:
    """Download (if needed) and load a frozen SimCLR ResNet18 checkpoint for
    the given model variant, returning a `FeatureBackbone` (an nn.Module)
    that yields the 6 tapped activations on `forward`.
    """
    if model_variant not in BACKBONE_REPOS:
        raise ValueError(f"Unknown model_variant {model_variant!r}. Options: {MODEL_VARIANTS}")

    repo_id = BACKBONE_REPOS[model_variant]
    files = list_repo_files(repo_id)
    # Most variants ship a .pth.tar; the ImageNet-1K/100 "reference" variants
    # ship a PyTorch-Lightning .ckpt instead (both are just torch.load-able
    # pickles under the hood).
    checkpoints = [f for f in files if f.endswith(".pth.tar") or f.endswith(".ckpt")]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth.tar/.ckpt checkpoint found in {repo_id}. Files: {files}")
    target = checkpoints[0]

    local_path = hf_hub_download(repo_id=repo_id, filename=target, local_dir=str(ASSETS_CACHE))

    try:
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model = SimCLRResNet18()
    is_lightning_seq_format = any(re.match(r"^backbone\.\d+\.", k) for k in state_dict)
    if is_lightning_seq_format:
        model.backbone.fc = nn.Identity()  # so backbone's state dict has no fc.* keys to match
        backbone_state = _remap_lightning_backbone_state_dict(state_dict)
        model.backbone.load_state_dict(backbone_state, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
        model.backbone.fc = nn.Identity()  # strip projection head -> 512-d features
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    extractor = FeatureBackbone(model.backbone)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


# ==============================================================================
# Component 2: bridging the undisclosed PCA step (documented approximation)
# ==============================================================================
def _get_projection_matrices() -> Dict[str, torch.Tensor]:
    """Load (or generate once and cache) the fixed seeded random orthogonal
    projection matrices used in place of the paper's undisclosed, unpublished
    PCA basis. Cached to assets_cache/random_projection_seed0.pt so the
    substitution is stable across runs/restarts.
    """
    if PROJECTION_CACHE_PATH.exists():
        return torch.load(PROJECTION_CACHE_PATH, weights_only=True)

    torch.manual_seed(0)
    mats: Dict[str, torch.Tensor] = {}
    for name in LAYER_KEYS:
        in_dim = PROJECTION_IN_DIMS[name]
        mat = torch.empty(in_dim, PROJECTION_OUT_DIM)
        nn.init.orthogonal_(mat)
        mats[name] = mat
    torch.save(mats, PROJECTION_CACHE_PATH)
    return mats


def project_to_1201(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    # APPROXIMATION: the published ridge weights were fit on 6 x 200-d PCA-reduced
    # features, but the fitted PCA basis was never publicly released (checked the
    # HF repo file tree and the companion github.com/DM-Diaz/eccentricity-constrained-simclr
    # repo this session -- it shows how PCA was fit, not the fitted object itself), so
    # each layer is instead global-average-pooled to its channel dimension and passed
    # through a fixed seeded random orthogonal projection to 200-d as a stand-in.
    """Reduce the 6 tapped activations to a differentiable 1201-d feature
    vector (6 x 200-d projected features + 1 constant intercept term), in the
    same layer order and total dimensionality the published ridge weights
    expect. This is a *substitute* for the paper's PCA step -- see the
    module-level comment above and the README's "Honest limitations" section.
    Kept fully differentiable (matmuls only, no numpy/.detach()) so gradients
    can flow all the way back to the phosphene stimulation image.
    """
    proj = _get_projection_matrices()
    parts = []
    for name in LAYER_KEYS:
        fmap = features[name]
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)  # (B, C)
        w = proj[name].to(device=pooled.device, dtype=pooled.dtype)
        parts.append(pooled @ w)  # (B, 200)
    feat_1200 = torch.cat(parts, dim=1)  # (B, 1200)
    ones = torch.ones(feat_1200.shape[0], 1, device=feat_1200.device, dtype=feat_1200.dtype)
    # Intercept appended last: the NSD weights matrix's final row (index 1200)
    # empirically has the smallest cross-voxel std of any row (checked this
    # session on subj01/Baseline), consistent with a separately-fit ridge
    # intercept term rather than one of the 6 x 200 PCA-derived feature rows.
    return torch.cat([feat_1200, ones], dim=1)  # (B, 1201)


# ==============================================================================
# Component 3: NSD ridge weights + V1 voxel selection
# ==============================================================================
@functools.lru_cache(maxsize=1)
def _nsd_repo_files() -> Tuple[str, ...]:
    return tuple(list_repo_files(NSD_REPO))


def filter_nsd_files(files, subject: str, model_variant: str) -> str:
    """Pick the one NSD encoding-model file matching `subject`/`model_variant`
    out of a repo file listing. Pure filtering logic (no network) so it's
    unit-testable against a fake file list -- see tests/test_pipeline.py.
    """
    if model_variant not in NSD_VARIANT_DIRS:
        raise ValueError(f"Unknown model_variant {model_variant!r}. Options: {MODEL_VARIANTS}")
    dir_prefix = NSD_VARIANT_DIRS[model_variant]
    pattern = re.compile(rf"^{re.escape(dir_prefix)}/NSD_{re.escape(subject)}_.*_concat\.npy$")
    matches = [f for f in files if pattern.match(f)]
    if not matches:
        raise FileNotFoundError(
            f"No NSD encoding-model file found for subject={subject!r}, "
            f"model_variant={model_variant!r} (looked for {pattern.pattern!r} in {NSD_REPO})"
        )
    return matches[0]


def _find_nsd_file(subject: str, model_variant: str) -> str:
    return filter_nsd_files(_nsd_repo_files(), subject, model_variant)


def _download_roi(subject_num: int) -> Path:
    cache_path = ASSETS_CACHE / f"prf-visualrois_subj{subject_num:02d}.nii.gz"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    url = (
        f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/"
        f"subj{subject_num:02d}/func1pt8mm/roi/prf-visualrois.nii.gz"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


@functools.lru_cache(maxsize=None)
def _load_nsd_fit(subject: str, model_variant: str) -> dict:
    """Download (if needed) and load the raw NSD ridge-regression fit dict
    for `subject`/`model_variant` (keys include `weights`, `voxel_index`,
    `brain_nii_shape` -- see module docstring history / README)."""
    filename = _find_nsd_file(subject, model_variant)
    local_path = hf_hub_download(repo_id=NSD_REPO, filename=filename, local_dir=str(ASSETS_CACHE))
    return np.load(local_path, allow_pickle=True).item()


def filter_v1_rows(roi_flat: np.ndarray, voxel_index: np.ndarray, labels=(1, 2)) -> np.ndarray:
    """Given a flattened ROI-atlas volume and the fit's flat voxel indices,
    return the positions *within* voxel_index (i.e. weight-matrix columns)
    whose ROI label is V1 (1=V1v, 2=V1d by default). Pure filtering logic
    (no network/no files) so it's unit-testable -- see tests/test_pipeline.py.
    """
    fitted_labels = roi_flat[voxel_index]
    return np.where(np.isin(fitted_labels, labels))[0]


@functools.lru_cache(maxsize=None)
def _v1_index_info(subject: str, model_variant: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Cross-reference the fit's flat voxel indices against the NSD visual-ROI
    atlas (labels 1=V1v, 2=V1d) to find which fitted voxels are V1.

    Returns (voxel_index [n_fitted,] flat C-order indices into the subject's
    (X,Y,Z) volume, v1_row_idx [n_v1,] positions *within* voxel_index/weight
    columns that are V1, brain_nii_shape (X,Y,Z)).
    """
    import nibabel as nib  # lazy import: optional dependency, only needed here

    d = _load_nsd_fit(subject, model_variant)
    voxel_index = np.asarray(d["voxel_index"][0])  # flat C-order indices, (n_voxels,)
    brain_nii_shape = tuple(int(x) for x in d["brain_nii_shape"])

    subject_num = int(subject[1:])
    roi_path = _download_roi(subject_num)
    roi = nib.load(str(roi_path)).get_fdata()
    roi_flat = roi.flatten(order="C")

    v1_row_idx = filter_v1_rows(roi_flat, voxel_index)
    return voxel_index, v1_row_idx, brain_nii_shape


@functools.lru_cache(maxsize=None)
def load_v1_weights(subject: str, model_variant: str) -> Tuple[np.ndarray, int]:
    """Restrict the frozen NSD ridge-regression weights for `subject`/
    `model_variant` to V1 voxels and return
    (v1_weight_matrix [1201, n_v1], n_v1_voxels).
    """
    d = _load_nsd_fit(subject, model_variant)
    weights = d["weights"]  # (1201, n_voxels) float64
    _, v1_row_idx, _ = _v1_index_info(subject, model_variant)
    if v1_row_idx.size == 0:
        raise RuntimeError(f"No V1 voxels found for subject={subject!r}, model_variant={model_variant!r}")

    v1_weights = weights[:, v1_row_idx].astype(np.float32)
    return v1_weights, int(v1_row_idx.size)


def _download_t1(subject_num: int) -> Path:
    cache_path = ASSETS_CACHE / f"T1_subj{subject_num:02d}.nii.gz"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    url = (
        f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/"
        f"subj{subject_num:02d}/func1pt8mm/T1_to_func1pt8mm.nii.gz"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


@functools.lru_cache(maxsize=None)
def load_brain_volumes(subject: str) -> Dict[str, np.ndarray]:
    """Real anatomical context for spatial visualization: a T1 anatomical
    volume and the NSD visual-ROI atlas, both already resampled by NSD into
    the subject's native func1pt8mm space -- the same voxel grid the
    ridge-regression fit and `v1_response_to_volume` use, so no further
    registration is needed. Returns {"t1", "roi" (both (X,Y,Z) arrays),
    "shape"}.
    """
    import nibabel as nib

    subject_num = int(subject[1:])
    t1_path = _download_t1(subject_num)
    roi_path = _download_roi(subject_num)
    t1 = np.asarray(nib.load(str(t1_path)).get_fdata(), dtype=np.float32)
    roi = np.asarray(nib.load(str(roi_path)).get_fdata(), dtype=np.int16)
    return {"t1": t1, "roi": roi, "shape": t1.shape}


def place_values_in_volume(
    v1_values: np.ndarray, voxel_index: np.ndarray, v1_row_idx: np.ndarray, brain_nii_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Scatter a (n_v1,) vector into a full (X,Y,Z) volume at the flat
    positions `voxel_index[v1_row_idx]`, NaN everywhere else. Pure array
    logic (no network/no files) so it's unit-testable -- see
    tests/test_pipeline.py.
    """
    v1_values = np.asarray(v1_values).reshape(-1)
    if v1_values.size != v1_row_idx.size:
        raise ValueError(f"Expected {v1_row_idx.size} V1 values, got {v1_values.size}")

    flat = np.full(int(np.prod(brain_nii_shape)), np.nan, dtype=np.float32)
    flat[voxel_index[v1_row_idx]] = v1_values.astype(np.float32)
    return flat.reshape(brain_nii_shape, order="C")


def v1_response_to_volume(subject: str, model_variant: str, v1_values: np.ndarray) -> np.ndarray:
    """Place a (n_v1,) vector -- in the same voxel order `load_v1_weights`/
    `extract_v1_target` use -- back into a full (X,Y,Z) volume in the
    subject's native func1pt8mm space, so it can be viewed as real brain
    slices. Voxels outside V1 are NaN.
    """
    voxel_index, v1_row_idx, brain_nii_shape = _v1_index_info(subject, model_variant)
    return place_values_in_volume(v1_values, voxel_index, v1_row_idx, brain_nii_shape)


# ==============================================================================
# Image preprocessing / V1 target extraction
# ==============================================================================
def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Resize/normalize a PIL image to the (1,3,224,224) ImageNet-normalized
    tensor the frozen backbone expects. Non-differentiable (PIL-based) --
    used only to ingest the initial real photo, not inside the optimization
    loop (see `_phosphene_to_backbone_input` for the differentiable version).
    """
    img = pil_image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (224,224,3) in [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _predict_v1(backbone: nn.Module, v1_weights_t: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
    """Shared differentiable path: image -> frozen backbone -> 1201-d
    projected features -> predicted V1 response (B, n_v1). No `torch.no_grad`
    here so this same function can be reused, with gradients flowing, inside
    the stimulation-inversion loop.
    """
    features = backbone(image_tensor)
    feat_1201 = project_to_1201(features)
    return feat_1201 @ v1_weights_t


def extract_v1_target(backbone: nn.Module, v1_weights: np.ndarray, image_tensor: torch.Tensor) -> torch.Tensor:
    """Run the real photo through the (differentiable) backbone -> projection
    -> V1-weights path, but detached, to get a fixed "target" V1 response to
    optimize the phosphene stimulation pattern against.
    """
    v1_weights_t = torch.from_numpy(v1_weights).to(dtype=image_tensor.dtype)
    with torch.no_grad():
        pred = _predict_v1(backbone, v1_weights_t, image_tensor)
    return pred.squeeze(0).detach()


# ==============================================================================
# Component 4: dynaphos phosphene simulator
# ==============================================================================
def build_simulator(n_electrodes_side: int, dropout: float, jitter: float) -> Tuple[GaussianSimulator, dict]:
    """Build a dynaphos GaussianSimulator on an n_electrodes_side x
    n_electrodes_side regular cortical electrode grid, with the given
    dropout fraction and positional-jitter scale. Returns (simulator, params).
    """
    params = load_params(str(CONFIG_PATH))
    params["cortex_model"]["dropout_rate"] = float(dropout)
    params["cortex_model"]["noise_scale"] = float(jitter)

    cortex_coords = get_cortex_coordinates_grid(params["cortex_model"], int(n_electrodes_side), int(n_electrodes_side))
    visual_field_coords = get_visual_field_coordinates_from_cortex_full(params["cortex_model"], cortex_coords)
    simulator = GaussianSimulator(params, visual_field_coords)
    return simulator, params


def _phosphene_to_backbone_input(phosphene: torch.Tensor) -> torch.Tensor:
    """Differentiably convert a (256,256) in-[0,1] phosphene image into a
    (1,3,224,224) ImageNet-normalized tensor for the frozen backbone. Uses
    torch ops only (no PIL/numpy) so autograd stays connected all the way
    back to the stimulation amplitude tensor.
    """
    x = phosphene.unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
    x = x.repeat(1, 3, 1, 1)  # (1,3,256,256)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def invert_to_stimulation(
    target_v1: torch.Tensor,
    backbone: nn.Module,
    v1_weights: np.ndarray,
    simulator: GaussianSimulator,
    params: dict,
    n_steps: int = 30,
    lr: float = 1e-5,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> dict:
    """Gradient-based inversion: search for a stimulation-current vector
    whose resulting phosphene percept, re-encoded through the frozen
    backbone + V1 weights, best matches `target_v1`.

    Returns {"final_phosphene": np.ndarray (256,256) in [0,1],
             "frames": list[np.ndarray] (a handful of intermediate snapshots),
             "loss_trace": list[float]}
    """
    target_v1 = target_v1.detach()
    v1_weights_t = torch.from_numpy(v1_weights).to(dtype=target_v1.dtype)

    # Seed the amplitude at a physically-plausible scale (Amperes) by sampling
    # a mid-gray activation image through the simulator's own stimulus-scale
    # conversion, rather than initializing raw current values from scratch.
    res_x, res_y = params["run"]["resolution"]
    init_image = torch.full((res_y, res_x), 0.5, dtype=target_v1.dtype)
    amplitude0 = simulator.sample_stimulus(init_image, rescale=True)
    # dynaphos's phosphene-size term is sqrt(amplitude / current_spread), which
    # has an infinite (NaN-producing) gradient exactly at amplitude == 0. A few
    # electrodes legitimately start at exactly 0 (degenerate/edge receptive
    # fields), so floor at a tiny positive epsilon -- physically still "no
    # negative current", but keeps every gradient finite.
    _MIN_AMPLITUDE = 1e-12
    amplitude0 = amplitude0.clamp(min=_MIN_AMPLITUDE)
    amplitude = amplitude0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([amplitude], lr=lr)

    n_steps = max(int(n_steps), 1)
    snapshot_every = max(1, n_steps // 8)

    frames = []
    loss_trace = []

    for step in range(n_steps):
        optimizer.zero_grad()
        simulator.reset()  # fresh, stateless render per optimization step
        phosphene = simulator(amplitude)  # (256,256) in [0,1], differentiable
        image_tensor = _phosphene_to_backbone_input(phosphene)
        pred_v1 = _predict_v1(backbone, v1_weights_t, image_tensor).squeeze(0)
        loss = F.mse_loss(pred_v1, target_v1)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            amplitude.clamp_(min=_MIN_AMPLITUDE)  # no negative current; keeps sqrt-size-term gradient finite

        loss_value = float(loss.item())
        loss_trace.append(loss_value)

        if step % snapshot_every == 0 or step == n_steps - 1:
            frames.append(phosphene.detach().cpu().numpy().copy())

        if progress_callback is not None:
            progress_callback(step + 1, n_steps, loss_value)

    with torch.no_grad():
        simulator.reset()
        final_phosphene = simulator(amplitude).detach().cpu().numpy()

    return {"final_phosphene": final_phosphene, "frames": frames, "loss_trace": loss_trace}


# ==============================================================================
# Smoke test
# ==============================================================================
if __name__ == "__main__":
    print("=== Phosphene Vision Pipeline: end-to-end smoke test ===")

    torch.manual_seed(123)
    random_img_arr = (torch.rand(224, 224, 3).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(random_img_arr, mode="RGB")

    t0 = time.time()
    backbone = load_backbone("VEDB Baseline")
    print(f"[1/6] load_backbone: OK ({time.time() - t0:.1f}s)")

    t0 = time.time()
    v1_weights, n_v1 = load_v1_weights("S1", "VEDB Baseline")
    print(f"[2/6] load_v1_weights: shape={v1_weights.shape} n_v1_voxels={n_v1} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    image_tensor = preprocess_image(pil_img)
    print(f"[3/6] preprocess_image: shape={tuple(image_tensor.shape)} dtype={image_tensor.dtype} ({time.time() - t0:.2f}s)")

    t0 = time.time()
    target_v1 = extract_v1_target(backbone, v1_weights, image_tensor)
    print(f"[4/6] extract_v1_target: shape={tuple(target_v1.shape)} ({time.time() - t0:.2f}s)")

    t0 = time.time()
    simulator, sim_params = build_simulator(n_electrodes_side=15, dropout=0.2, jitter=0.4)
    print(f"[5/6] build_simulator: num_phosphenes={simulator.num_phosphenes} shape={simulator.shape} ({time.time() - t0:.2f}s)")

    t0 = time.time()
    result = invert_to_stimulation(
        target_v1,
        backbone,
        v1_weights,
        simulator,
        sim_params,
        n_steps=15,
        lr=1e-5,
        progress_callback=lambda i, n, l: print(f"      step {i}/{n}  loss={l:.6f}"),
    )
    print(f"[6/6] invert_to_stimulation: done ({time.time() - t0:.1f}s)")

    print("\n--- Results ---")
    print("final_phosphene shape:", result["final_phosphene"].shape, result["final_phosphene"].dtype)
    print("final_phosphene min/max:", float(result["final_phosphene"].min()), float(result["final_phosphene"].max()))
    print("num frames captured:", len(result["frames"]))
    print("loss trace:", result["loss_trace"])
    print("\nSmoke test PASSED.")
