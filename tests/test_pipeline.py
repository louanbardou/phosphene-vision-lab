"""Unit tests for pipeline.py's data import / filtering functions.

These deliberately avoid the network: no HuggingFace downloads, no NSD S3
requests. They exercise the pure filtering/placement logic on small,
synthetic inputs, so the suite stays fast and reliable in CI.
"""

import numpy as np
import pytest
from PIL import Image

import pipeline as p


# --------------------------------------------------------------------------
# filter_nsd_files: picking the right NSD encoding-model file out of a
# repo file listing (the "import the right data file" logic).
# --------------------------------------------------------------------------
FAKE_NSD_FILES = [
    "baseline/NSD_S1_resnet18-Baseline_concat.npy",
    "baseline/NSD_S2_resnet18-Baseline_concat.npy",
    "fovea-gaze/NSD_S1_resnet18-Fovea-Gaze_concat.npy",
    "periph/NSD_S1_resnet18-Periph_concat.npy",
    "periph-nf/NSD_S1_resnet18-Periph-NF_concat.npy",
    "reference-models/resnet18-simclr-imagenet1k/NSD_S1_resnet18-simclr-imagenet1k_concat.npy",
]


def test_filter_nsd_files_finds_exact_match():
    result = p.filter_nsd_files(FAKE_NSD_FILES, "S1", "VEDB Baseline")
    assert result == "baseline/NSD_S1_resnet18-Baseline_concat.npy"


def test_filter_nsd_files_does_not_confuse_periph_and_periph_nf():
    # "periph" must not accidentally match "periph-nf/..." or vice versa.
    result = p.filter_nsd_files(FAKE_NSD_FILES, "S1", "VEDB Periph")
    assert result == "periph/NSD_S1_resnet18-Periph_concat.npy"

    result_nf = p.filter_nsd_files(FAKE_NSD_FILES, "S1", "VEDB Periph-NF")
    assert result_nf == "periph-nf/NSD_S1_resnet18-Periph-NF_concat.npy"


def test_filter_nsd_files_respects_subject():
    result = p.filter_nsd_files(FAKE_NSD_FILES, "S2", "VEDB Baseline")
    assert "NSD_S2_" in result


def test_filter_nsd_files_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        p.filter_nsd_files(FAKE_NSD_FILES, "S8", "VEDB Baseline")  # not in the fake listing


def test_filter_nsd_files_raises_on_unknown_model_variant():
    with pytest.raises(ValueError):
        p.filter_nsd_files(FAKE_NSD_FILES, "S1", "Not A Real Variant")


# --------------------------------------------------------------------------
# filter_v1_rows: selecting V1 voxels (labels 1=V1v, 2=V1d) among the
# fitted voxels, from a synthetic ROI atlas.
# --------------------------------------------------------------------------
def test_filter_v1_rows_selects_only_v1_labels():
    # A tiny fake "brain volume" of ROI labels, flattened.
    roi_flat = np.array([-1, 0, 1, 2, 3, 4, 0, 1, 5, 2])
    # The fit only kept these 6 voxels (by their flat index into roi_flat).
    voxel_index = np.array([1, 2, 3, 4, 7, 9])
    v1_row_idx = p.filter_v1_rows(roi_flat, voxel_index)

    # Expected: positions within voxel_index whose roi label is 1 or 2.
    # voxel_index -> labels: [1]->0, [2]->1, [3]->2, [4]->3, [7]->1, [9]->2
    # V1 (label 1 or 2) at positions 1, 2, 4, 5.
    assert list(v1_row_idx) == [1, 2, 4, 5]


def test_filter_v1_rows_empty_when_no_v1_voxels():
    roi_flat = np.array([0, 3, 4, 5])
    voxel_index = np.array([0, 1, 2, 3])
    v1_row_idx = p.filter_v1_rows(roi_flat, voxel_index)
    assert v1_row_idx.size == 0


def test_filter_v1_rows_custom_labels():
    roi_flat = np.array([3, 4, 5, 6, 7])
    voxel_index = np.array([0, 1, 2, 3, 4])
    v1_row_idx = p.filter_v1_rows(roi_flat, voxel_index, labels=(5, 6))
    assert list(v1_row_idx) == [2, 3]


# --------------------------------------------------------------------------
# place_values_in_volume: scattering a per-V1-voxel vector back into a
# full anatomical volume.
# --------------------------------------------------------------------------
def test_place_values_in_volume_shape_and_placement():
    brain_shape = (2, 2, 2)  # 8 voxels total
    voxel_index = np.array([0, 3, 5])  # 3 fitted voxels
    v1_row_idx = np.array([1, 2])  # of those, indices 1 and 2 are V1 -> flat positions 3, 5
    values = np.array([10.0, 20.0], dtype=np.float32)

    vol = p.place_values_in_volume(values, voxel_index, v1_row_idx, brain_shape)

    assert vol.shape == brain_shape
    flat = vol.reshape(-1)
    assert flat[3] == pytest.approx(10.0)
    assert flat[5] == pytest.approx(20.0)
    # every other voxel should be NaN
    other_positions = [i for i in range(8) if i not in (3, 5)]
    assert np.all(np.isnan(flat[other_positions]))


def test_place_values_in_volume_raises_on_size_mismatch():
    with pytest.raises(ValueError):
        p.place_values_in_volume(
            v1_values=np.array([1.0, 2.0, 3.0]),
            voxel_index=np.array([0, 1]),
            v1_row_idx=np.array([0, 1]),  # expects 2 values, got 3
            brain_nii_shape=(2, 2, 1),
        )


# --------------------------------------------------------------------------
# preprocess_image: photo -> normalized tensor the backbone expects.
# --------------------------------------------------------------------------
def test_preprocess_image_shape_and_dtype():
    img = Image.new("RGB", (64, 128), color=(128, 64, 32))
    tensor = p.preprocess_image(img)
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert tensor.dtype.__str__().endswith("float32")


def test_preprocess_image_is_imagenet_normalized():
    # A flat mid-gray image should land close to 0 after ImageNet
    # normalization (mean ~0.45..0.48, so 0.5-centered input ~ near zero).
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    tensor = p.preprocess_image(img)
    assert float(tensor.mean().abs()) < 0.5


# --------------------------------------------------------------------------
# Config sanity: every model variant must be wired consistently across the
# lookup tables the rest of the pipeline relies on. Catches typos that would
# otherwise only surface as a runtime KeyError deep in a download call.
# --------------------------------------------------------------------------
def test_model_variant_tables_are_consistent():
    assert set(p.MODEL_VARIANTS) == set(p.BACKBONE_REPOS.keys())
    assert set(p.MODEL_VARIANTS) == set(p.NSD_VARIANT_DIRS.keys())


def test_subjects_are_well_formed():
    for subject in p.SUBJECTS:
        assert subject[0] == "S"
        assert subject[1:].isdigit()
