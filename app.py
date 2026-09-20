"""
Phosphene Vision Pipeline — Streamlit UI

A real (not mocked) research pipeline:
  photo -> frozen CNN feature extraction -> frozen NSD ridge-regression V1
  encoding weights -> gradient-based optimization of a phosphene stimulation
  pattern -> a differentiable cortical phosphene simulator (dynaphos) ->
  a simulated percept for a cortical visual prosthesis.

This file is the UI layer only. All model/science logic lives in pipeline.py
(built in parallel by another agent) and is imported below.
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Pipeline import (guarded so the UI shell still loads if pipeline.py is not
# present yet / not finished — no fake pipeline logic is stubbed in here).
# --------------------------------------------------------------------------
try:
    from pipeline import (
        SUBJECTS,
        MODEL_VARIANTS,
        load_backbone,
        load_v1_weights,
        load_brain_volumes,
        v1_response_to_volume,
        preprocess_image,
        extract_v1_target,
        build_simulator,
        invert_to_stimulation,
    )

    PIPELINE_AVAILABLE = True
    PIPELINE_IMPORT_ERROR = None
except ImportError as exc:  # pipeline.py missing or incomplete
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(exc)
    SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    MODEL_VARIANTS = [
        "VEDB Baseline",
        "VEDB Fovea-Gaze",
        "VEDB Periph",
        "VEDB Periph-NF",
        "ImageNet-1K (reference)",
        "ImageNet-100 (reference)",
    ]


# ==========================================================================
# Page config
# ==========================================================================
st.set_page_config(
    page_title="Phosphene Vision Pipeline",
    page_icon="\U0001F441️",  # eye emoji
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Custom CSS — dark, polished, CNN-Explainer-inspired panel layout
# ==========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% -10%, #1b2333 0%, #0e1117 45%, #0a0c10 100%);
        color: #e6e8ee;
    }

    section[data-testid="stSidebar"] {
        background: #12151d;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }

    /* Header banner */
    .pv-header {
        background: linear-gradient(120deg, #1c2534 0%, #22304a 55%, #1a2740 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }
    .pv-header h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
        color: #f4f6fb;
    }
    .pv-header p {
        margin: 0;
        color: #a9b2c6;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 62rem;
    }
    .pv-header .pv-tag {
        display: inline-block;
        margin-top: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: #9db4ff;
        background: rgba(76, 125, 255, 0.10);
        border: 1px solid rgba(76, 125, 255, 0.30);
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
    }

    /* Step-flow strip */
    .pv-flow {
        display: flex;
        align-items: stretch;
        gap: 0.4rem;
        margin: 0.2rem 0 1.3rem 0;
        flex-wrap: wrap;
    }
    .pv-flow-step {
        flex: 1 1 150px;
        background: #151a24;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        min-width: 140px;
    }
    .pv-flow-step .pv-flow-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #4C7DFF;
        color: #0a0c10;
        font-weight: 700;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
    }
    .pv-flow-step .pv-flow-title {
        font-weight: 600;
        font-size: 0.86rem;
        color: #e6e8ee;
        margin-bottom: 0.15rem;
    }
    .pv-flow-step .pv-flow-desc {
        font-size: 0.74rem;
        color: #8b93a7;
        line-height: 1.35;
    }
    .pv-flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #4a5268;
        font-size: 1.1rem;
        flex: 0 0 auto;
        padding: 0 0.1rem;
    }

    /* Panel headers inside result cards */
    .pv-panel-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f0f2f7;
        margin-bottom: 0.15rem;
    }
    .pv-panel-caption {
        font-size: 0.78rem;
        color: #8b93a7;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }
    .pv-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 6px;
        background: rgba(76, 125, 255, 0.18);
        color: #9db4ff;
        font-weight: 700;
        font-size: 0.72rem;
        margin-right: 0.4rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
    }

    hr { border-color: rgba(255,255,255,0.08); }

    .pv-footnote {
        font-size: 0.75rem;
        color: #6b7284;
        margin-top: 0.3rem;
    }

    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Cached loaders (app-level caching per the pipeline contract)
# ==========================================================================
if PIPELINE_AVAILABLE:

    @st.cache_resource(show_spinner=False)
    def cached_load_backbone(model_variant: str):
        return load_backbone(model_variant)

    @st.cache_resource(show_spinner=False)
    def cached_load_v1_weights(subject: str, model_variant: str):
        return load_v1_weights(subject, model_variant)

    @st.cache_resource(show_spinner=False)
    def cached_load_brain_volumes(subject: str):
        return load_brain_volumes(subject)


# ==========================================================================
# Small rendering helpers (dependency-free: numpy + PIL only)
# ==========================================================================
_CMAP_ANCHORS = np.array(
    [
        [30, 20, 60],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ],
    dtype=np.float64,
)


def _apply_colormap(norm: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Map a [0,1]-normalized 2D array to an RGB image with a viridis-like
    hand-rolled colormap (avoids a matplotlib dependency)."""
    flat = np.nan_to_num(norm, nan=0.0).ravel()
    positions = np.linspace(0.0, 1.0, _CMAP_ANCHORS.shape[0])
    r = np.interp(flat, positions, _CMAP_ANCHORS[:, 0])
    g = np.interp(flat, positions, _CMAP_ANCHORS[:, 1])
    b = np.interp(flat, positions, _CMAP_ANCHORS[:, 2])
    rgb = np.stack([r, g, b], axis=-1).reshape(norm.shape + (3,))
    rgb[nan_mask] = (15, 17, 23)
    return rgb.astype(np.uint8)


AXIS_OPTIONS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def render_brain_slice(
    t1: np.ndarray, v1_volume: np.ndarray, axis: int, index: int,
    overlay_alpha: float = 0.75, upscale: int = 4,
) -> Image.Image:
    """Render one real anatomical slice (T1, subject-native func1pt8mm space)
    with the predicted V1 response overlaid in color wherever a V1 voxel
    exists on that slice. `axis` is 0=sagittal, 1=coronal, 2=axial."""

    def _take(vol: np.ndarray) -> np.ndarray:
        if axis == 0:
            return vol[index, :, :]
        if axis == 1:
            return vol[:, index, :]
        return vol[:, :, index]

    t1_slice = _take(t1)
    v1_slice = _take(v1_volume)

    finite_t1 = t1_slice[np.isfinite(t1_slice)]
    lo, hi = (np.percentile(finite_t1, 1), np.percentile(finite_t1, 99)) if finite_t1.size else (0.0, 1.0)
    hi = hi if hi > lo else lo + 1.0
    gray = np.clip((t1_slice - lo) / (hi - lo), 0.0, 1.0)
    base_rgb = np.stack([gray, gray, gray], axis=-1)

    v1_mask = np.isfinite(v1_slice)
    out = base_rgb
    if v1_mask.any():
        finite_v1 = v1_slice[v1_mask]
        vmin, vmax = float(finite_v1.min()), float(finite_v1.max())
        rng = (vmax - vmin) if vmax > vmin else 1.0
        norm = np.clip((v1_slice - vmin) / rng, 0.0, 1.0)
        overlay_rgb = _apply_colormap(norm, ~v1_mask).astype(np.float64) / 255.0
        out = base_rgb.copy()
        out[v1_mask] = (1 - overlay_alpha) * base_rgb[v1_mask] + overlay_alpha * overlay_rgb[v1_mask]

    arr = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
    arr = np.rot90(arr)  # NSD volumes are stored so a raw slice reads sideways
    img = Image.fromarray(arr, mode="RGB")
    return img.resize((img.width * upscale, img.height * upscale), resample=Image.NEAREST)


def render_phosphene_with_axes(
    phosphene: np.ndarray, sim_params: dict, upscale: int = 3, n_ticks: int = 5,
) -> Image.Image:
    """Render a phosphene percept anchored in the real visual-field
    coordinates dynaphos itself simulates in (degrees of visual angle,
    origin = fixation), instead of an unlabeled black canvas — the same
    axes convention used in the dynaphos paper's own figures."""
    res_x, res_y = sim_params["run"]["resolution"]
    view_angle = float(sim_params["run"]["view_angle"])
    origin_x, origin_y = sim_params["run"]["origin"]
    hemi = view_angle / 2.0
    x_min, x_max = origin_x - hemi, origin_x + hemi
    y_min, y_max = origin_y - hemi, origin_y + hemi

    gray = np.clip(np.asarray(phosphene, dtype=np.float64), 0.0, 1.0)
    img_w, img_h = int(res_x * upscale), int(res_y * upscale)
    base = Image.fromarray((gray * 255).astype(np.uint8), mode="L").convert("RGB")
    base = base.resize((img_w, img_h), resample=Image.NEAREST)

    margin_l, margin_b, margin_t, margin_r = 40, 34, 10, 10
    canvas = Image.new("RGB", (img_w + margin_l + margin_r, img_h + margin_t + margin_b), (10, 12, 16))
    canvas.paste(base, (margin_l, margin_t))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    tick_color = (139, 147, 167)
    border_color = (60, 66, 82)

    draw.rectangle(
        [margin_l, margin_t, margin_l + img_w - 1, margin_t + img_h - 1],
        outline=border_color, width=1,
    )

    for tv in np.linspace(x_min, x_max, n_ticks):
        frac = (tv - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        px = margin_l + frac * (img_w - 1)
        draw.line([(px, margin_t + img_h), (px, margin_t + img_h + 4)], fill=tick_color, width=1)
        draw.text((px - 7, margin_t + img_h + 6), f"{tv:.0f}°", fill=tick_color, font=font)

    for tv in np.linspace(y_min, y_max, n_ticks):
        frac = (tv - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        py = margin_t + img_h - frac * (img_h - 1)  # up = +elevation
        draw.line([(margin_l - 4, py), (margin_l, py)], fill=tick_color, width=1)
        draw.text((2, py - 5), f"{tv:.0f}°", fill=tick_color, font=font)

    ox_frac = (origin_x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
    oy_frac = (origin_y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
    ox_px = margin_l + ox_frac * (img_w - 1)
    oy_px = margin_t + img_h - oy_frac * (img_h - 1)
    r = 4
    cross_color = (76, 125, 255)
    draw.line([(ox_px - r, oy_px), (ox_px + r, oy_px)], fill=cross_color, width=1)
    draw.line([(ox_px, oy_px - r), (ox_px, oy_px + r)], fill=cross_color, width=1)

    caption = "azimuth / elevation, deg. visual angle · + = fixation"
    draw.text((margin_l, margin_t + img_h + 18), caption, fill=tick_color, font=font)

    return canvas


def to_numpy(x):
    """Best-effort tensor/array -> numpy conversion without importing torch
    directly in the UI layer."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def render_electrode_grid_preview(
    n_side: int, dropout: float, jitter: float, size: int = 200
) -> Image.Image:
    """Cheap, dependency-free schematic of the assumed regular electrode
    grid (not a measured implant) — safe to redraw on every slider tweak."""
    n_side = max(int(n_side), 1)
    rng = np.random.default_rng(seed=1000 + n_side)
    canvas = Image.new("RGB", (size, size), (16, 19, 26))
    draw = ImageDraw.Draw(canvas)
    margin = size * 0.10
    usable = size - 2 * margin
    spacing = usable / max(n_side - 1, 1) if n_side > 1 else 0.0
    dot_r = max(1.6, spacing * 0.16) if n_side > 1 else size * 0.05
    max_jitter_px = spacing * 0.5 * jitter if n_side > 1 else 0.0

    for i in range(n_side):
        for j in range(n_side):
            if rng.random() < dropout:
                continue
            x = margin + j * spacing + rng.uniform(-max_jitter_px, max_jitter_px)
            y = margin + i * spacing + rng.uniform(-max_jitter_px, max_jitter_px)
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=(76, 125, 255))

    return canvas


if not PIPELINE_AVAILABLE:
    st.error(
        "pipeline.py could not be imported, so the pipeline cannot run yet. "
        f"Import error: `{PIPELINE_IMPORT_ERROR}`. The UI below still renders "
        "so layout/styling can be reviewed independently."
    )

# Header + step-flow strip only shown before a result exists — once you've
# run the pipeline, the results panel below is the point, not the intro.
if "pv_result" not in st.session_state:
    # ----------------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------------
    st.markdown(
        """
        <div class="pv-header">
            <h1>Phosphene Vision Pipeline</h1>
            <p>
                Turns a single photo into a simulated cortical-prosthesis percept:
                a frozen self-supervised CNN extracts features, a frozen NSD
                ridge-regression fit predicts a target V1 response, a gradient-based
                optimizer searches for the electrode stimulation pattern that best
                reproduces that response, and a differentiable phosphene simulator
                (dynaphos) renders what the resulting percept might look like.
            </p>
            <span class="pv-tag">real weights · real optimization · not real-time</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step-flow strip (static, explanatory — CNN-Explainer-style panel-per-stage)
    _flow_steps = [
        ("1", "Input photo", "A snapshot or uploaded image."),
        ("2", "Frozen CNN", "SimCLR ResNet18 features, weights untouched."),
        ("3", "V1 encoding", "Per-subject NSD ridge-regression fit predicts a target V1 response."),
        ("4", "Optimize stimulation", "Gradient descent searches for an electrode pattern matching that target."),
        ("5", "Phosphene simulator", "dynaphos renders the resulting simulated percept — the pattern of light the wearer would subjectively see."),
    ]
    _flow_html = ['<div class="pv-flow">']
    for i, (num, title, desc) in enumerate(_flow_steps):
        _flow_html.append(
            f'<div class="pv-flow-step"><span class="pv-flow-num">{num}</span>'
            f'<div class="pv-flow-title">{title}</div>'
            f'<div class="pv-flow-desc">{desc}</div></div>'
        )
        if i < len(_flow_steps) - 1:
            _flow_html.append('<div class="pv-flow-arrow">&#8594;</div>')
    _flow_html.append("</div>")
    st.markdown("".join(_flow_html), unsafe_allow_html=True)



# ==========================================================================
# Sidebar controls
# ==========================================================================
st.sidebar.markdown("### Input image")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

raw_file = uploaded_file
input_image = None
if raw_file is not None:
    try:
        input_image = Image.open(raw_file).convert("RGB")
    except Exception as exc:
        st.sidebar.error(f"Could not read image: {exc}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Subject & model")
subject = st.sidebar.selectbox("Subject (NSD ridge-regression fit)", SUBJECTS, index=0)
_default_model_idx = (
    MODEL_VARIANTS.index("VEDB Baseline") if "VEDB Baseline" in MODEL_VARIANTS else 0
)
model_variant = st.sidebar.selectbox(
    "Backbone / model variant", MODEL_VARIANTS, index=_default_model_idx
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Electrode array")
n_electrodes_side = st.sidebar.slider("Electrodes per side", min_value=5, max_value=25, value=15, step=1)
dropout = st.sidebar.slider(
    "Simulated electrode dropout", min_value=0.0, max_value=0.5, value=0.2, step=0.01,
    help="Fraction of electrodes assumed non-functional / dropped out.",
)
jitter = st.sidebar.slider(
    "Electrode position jitter", min_value=0.0, max_value=1.0, value=0.4, step=0.01,
    help="Random displacement applied to each electrode's assumed cortical position.",
)

st.sidebar.caption("Electrode grid preview")
st.sidebar.image(
    render_electrode_grid_preview(n_electrodes_side, dropout, jitter),
    use_container_width=False,
    width=170,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Optimization")
n_steps = st.sidebar.slider("Optimization steps", min_value=10, max_value=300, value=120, step=10)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button(
    "Optimize stimulation pattern", type="primary", use_container_width=True
)


# ==========================================================================
# Run pipeline on button click
# ==========================================================================
if run_clicked:
    if input_image is None:
        st.sidebar.error("Upload an image first.")
    elif not PIPELINE_AVAILABLE:
        st.error(f"Cannot run: pipeline.py is unavailable ({PIPELINE_IMPORT_ERROR}).")
    else:
        try:
            with st.spinner(f"Loading {model_variant} backbone..."):
                backbone = cached_load_backbone(model_variant)

            with st.spinner(f"Loading NSD V1 ridge-regression weights for subject {subject}..."):
                v1_weights, n_v1_voxels = cached_load_v1_weights(subject, model_variant)

            with st.spinner("Extracting frozen-CNN features and predicting the V1 target response..."):
                image_tensor = preprocess_image(input_image)
                v1_target = extract_v1_target(backbone, v1_weights, image_tensor)

            with st.spinner(f"Loading subject {subject}'s real anatomical brain volume..."):
                brain = cached_load_brain_volumes(subject)
                v1_volume = v1_response_to_volume(subject, model_variant, to_numpy(v1_target))

            with st.spinner("Building the cortical phosphene simulator..."):
                simulator, sim_params = build_simulator(n_electrodes_side, dropout, jitter)

            progress_bar = st.progress(0, text="Optimizing stimulation pattern...")

            def _progress_cb(*args, **kwargs):
                step = kwargs.get("step", args[0] if len(args) > 0 else None)
                total = kwargs.get("total", args[1] if len(args) > 1 else n_steps)
                loss = kwargs.get("loss", args[2] if len(args) > 2 else None)
                try:
                    frac = float(step) / float(total) if step is not None and total else 0.0
                except (TypeError, ZeroDivisionError):
                    frac = 0.0
                frac = min(max(frac, 0.0), 1.0)
                msg = "Optimizing stimulation pattern..."
                if step is not None and total:
                    try:
                        msg = f"Optimization step {int(step)}/{int(total)}"
                    except (TypeError, ValueError):
                        pass
                if loss is not None:
                    try:
                        msg += f" — loss {float(loss):.4f}"
                    except (TypeError, ValueError):
                        pass
                progress_bar.progress(frac, text=msg)

            result = invert_to_stimulation(
                v1_target,
                backbone,
                v1_weights,
                simulator,
                sim_params,
                n_steps=n_steps,
                lr=3e-4,
                progress_callback=_progress_cb,
            )
            progress_bar.progress(1.0, text="Optimization complete.")

            st.session_state["pv_result"] = result
            st.session_state["pv_input_image"] = input_image
            st.session_state["pv_v1_target"] = to_numpy(v1_target)
            st.session_state["pv_brain"] = brain
            st.session_state["pv_v1_volume"] = v1_volume
            st.session_state["pv_sim_params"] = sim_params
            st.session_state["pv_meta"] = dict(
                subject=subject,
                model_variant=model_variant,
                n_electrodes_side=n_electrodes_side,
                dropout=dropout,
                jitter=jitter,
                n_steps=n_steps,
                n_v1_voxels=n_v1_voxels,
            )
        except Exception as exc:
            st.error(f"Pipeline run failed: {exc}")
            with st.expander("Traceback (debug)"):
                st.exception(exc)


# ==========================================================================
# Results
# ==========================================================================
st.markdown("---")

if "pv_result" in st.session_state:
    result = st.session_state["pv_result"]
    meta = st.session_state["pv_meta"]
    result_image = st.session_state["pv_input_image"]
    sim_params = st.session_state["pv_sim_params"]

    final_phosphene = np.asarray(result.get("final_phosphene"))
    frames = result.get("frames") or []
    loss_trace = result.get("loss_trace") or []

    st.markdown("## Pipeline output")
    st.caption(
        f"Subject {meta['subject']} · {meta['model_variant']} · "
        f"{meta['n_electrodes_side']}×{meta['n_electrodes_side']} electrode grid · "
        f"dropout {meta['dropout']:.2f} · jitter {meta['jitter']:.2f} · "
        f"{meta['n_steps']} optimization steps · {meta['n_v1_voxels']} V1 voxels"
    )

    col1, col2, col3, col4 = st.columns(4)
    PANEL_HEIGHT = 620

    with col1:
        with st.container(border=True, height=PANEL_HEIGHT):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">1</span>Input photo</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">The photo fed into the frozen CNN.</div>',
                unsafe_allow_html=True,
            )
            st.image(result_image, use_container_width=True)

    with col2:
        with st.container(border=True, height=PANEL_HEIGHT):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">2</span>V1 response on the brain</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">Predicted V1 activity placed back onto '
                f"subject {meta['subject']}'s real anatomy (NSD T1, native func1pt8mm "
                "space) — color = predicted response where a V1 voxel exists on this "
                "slice, grayscale = anatomy elsewhere.</div>",
                unsafe_allow_html=True,
            )
            brain = st.session_state.get("pv_brain")
            v1_volume = st.session_state.get("pv_v1_volume")
            if brain is not None and v1_volume is not None:
                axis_label = st.radio(
                    "View", list(AXIS_OPTIONS.keys()), index=2, horizontal=True, key="pv_axis_label",
                )
                axis = AXIS_OPTIONS[axis_label]
                max_idx = brain["shape"][axis] - 1
                slice_idx = st.slider(
                    "Slice", min_value=0, max_value=max_idx, value=max_idx // 2, key=f"pv_slice_{axis_label}",
                )
                st.image(
                    render_brain_slice(brain["t1"], v1_volume, axis, slice_idx),
                    use_container_width=True,
                )
            else:
                st.info("Run the pipeline to see this.")

    with col3:
        with st.container(border=True, height=PANEL_HEIGHT):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">3</span>Simulated percept</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">Final phosphene pattern from the dynaphos '
                "simulator after optimization, anchored in degrees of visual angle.</div>",
                unsafe_allow_html=True,
            )
            st.image(render_phosphene_with_axes(final_phosphene, sim_params), use_container_width=True)

    with col4:
        with st.container(border=True, height=PANEL_HEIGHT):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">4</span>Optimization scrubber</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">Step through intermediate snapshots to see the '
                "percept sharpen.</div>",
                unsafe_allow_html=True,
            )
            if frames:
                frame_idx = st.select_slider(
                    "Frame",
                    options=list(range(len(frames))),
                    value=len(frames) - 1,
                    format_func=lambda i: f"step {i + 1}/{len(frames)}",
                    label_visibility="collapsed",
                )
                st.image(
                    render_phosphene_with_axes(np.asarray(frames[frame_idx]), sim_params),
                    use_container_width=True,
                )
            else:
                st.info("No intermediate frames were returned by the optimizer.")

    st.markdown("#### Optimization loss")
    if loss_trace:
        st.line_chart(loss_trace)
        st.markdown(
            '<div class="pv-footnote">Loss between the simulator\'s predicted V1 response '
            "for the current stimulation pattern and the target V1 response, per step.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No loss trace was returned by the optimizer.")

else:
    st.info(
        "Upload a photo in the sidebar, choose a subject and model variant, "
        "then click **Optimize stimulation pattern** to run the pipeline."
    )
