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
from PIL import Image, ImageDraw

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
        color: #8fd6c4;
        background: rgba(94, 201, 170, 0.10);
        border: 1px solid rgba(94, 201, 170, 0.30);
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
        background: #5ec9aa;
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

    /* Caveats callout */
    .pv-caveat-box {
        border-left: 3px solid #d9a441;
        background: rgba(217, 164, 65, 0.07);
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.4rem;
    }
    .pv-caveat-box h4 {
        margin: 0 0 0.3rem 0;
        color: #e8c47c;
        font-size: 0.92rem;
        font-weight: 700;
    }
    .pv-caveat-box p, .pv-caveat-box li {
        color: #c7cbd6;
        font-size: 0.87rem;
        line-height: 1.55;
    }
    .pv-caveat-box ol {
        margin: 0.3rem 0 0.2rem 1.1rem;
        padding: 0;
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
        background: rgba(94, 201, 170, 0.18);
        color: #5ec9aa;
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


def render_vector_heatmap(vector: np.ndarray, cell_px: int = 9) -> Image.Image:
    """Render a 1D vector (e.g. ~1000+ V1 target values) as a compact 2D
    heatmap strip, roughly square, rather than literal brain anatomy."""
    v = np.asarray(vector, dtype=np.float64).ravel()
    n = max(v.size, 1)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    padded = np.full(rows * cols, np.nan)
    padded[:n] = v
    grid = padded.reshape(rows, cols)

    finite = v[np.isfinite(v)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    rng = (vmax - vmin) if vmax > vmin else 1.0
    norm = np.clip((grid - vmin) / rng, 0.0, 1.0)
    nan_mask = np.isnan(grid)

    rgb = _apply_colormap(norm, nan_mask)
    img = Image.fromarray(rgb, mode="RGB")
    target_w = min(cols * cell_px, 420)
    scale = max(1, target_w // max(cols, 1))
    img = img.resize((cols * scale, rows * scale), resample=Image.NEAREST)
    return img


def array_to_gray_image(arr: np.ndarray) -> Image.Image:
    """Render a [0,1] float array as an 8-bit grayscale image."""
    a = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    return Image.fromarray((a * 255).astype(np.uint8), mode="L")


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
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=(94, 201, 170))

    return canvas


# ==========================================================================
# Header
# ==========================================================================
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

if not PIPELINE_AVAILABLE:
    st.error(
        "pipeline.py could not be imported, so the pipeline cannot run yet. "
        f"Import error: `{PIPELINE_IMPORT_ERROR}`. The UI below still renders "
        "so layout/styling can be reviewed independently."
    )

# Step-flow strip (static, explanatory — CNN-Explainer-style panel-per-stage)
_flow_steps = [
    ("1", "Input photo", "A snapshot or uploaded image."),
    ("2", "Frozen CNN", "SimCLR ResNet18 features, weights untouched."),
    ("3", "V1 encoding", "Per-subject NSD ridge-regression fit predicts a target V1 response."),
    ("4", "Optimize stimulation", "Gradient descent searches for an electrode pattern matching that target."),
    ("5", "Phosphene simulator", "dynaphos renders the resulting simulated percept."),
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
# Caveats — permanent, visible, open by default
# ==========================================================================
with st.expander("Read before you trust this", expanded=True):
    st.markdown(
        """
        <div class="pv-caveat-box">
        <h4>Three simplifications made in this demo</h4>
        <ol>
        <li><b>Feature reduction is substituted.</b> The published PCA step used
        by the original NSD encoding-model authors to reduce CNN features before
        ridge regression is not publicly distributed. This demo substitutes a
        fixed random projection in its place. Treat the "V1 target response"
        panel as illustrative of the pipeline mechanism, not a faithful
        reproduction of the paper's exact voxel predictions.</li>
        <li><b>Subject weights are not yours.</b> The NSD ridge-regression
        weights are fit per subject (S1&ndash;S8) on brain data collected from
        that specific person. Applying them to an arbitrary photo of your own
        choosing is an approximation of the pipeline mechanism, not a
        personalized reading of your own visual cortex.</li>
        <li><b>The electrode grid is assumed, not measured.</b> The electrode
        layout is a regular grid placed over a cortical map as a modeling
        convenience, not a real or measured electrode placement from any
        implant.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Sidebar controls
# ==========================================================================
st.sidebar.markdown("### Input image")
image_source = st.sidebar.radio(
    "Image source", ["Upload a photo", "Camera snapshot"], index=0, label_visibility="collapsed"
)

uploaded_file = None
camera_file = None
if image_source == "Upload a photo":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
else:
    camera_file = st.sidebar.camera_input("Take a snapshot", label_visibility="collapsed")

raw_file = uploaded_file if uploaded_file is not None else camera_file
input_image = None
if raw_file is not None:
    try:
        input_image = Image.open(raw_file).convert("RGB")
    except Exception as exc:
        st.sidebar.error(f"Could not read image: {exc}")

if input_image is not None:
    st.sidebar.image(input_image, caption="Selected input", use_container_width=True)

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

st.sidebar.caption("Assumed electrode grid preview (schematic, not a measured implant)")
st.sidebar.image(
    render_electrode_grid_preview(n_electrodes_side, dropout, jitter),
    use_container_width=False,
    width=170,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Optimization")
n_steps = st.sidebar.slider("Optimization steps", min_value=10, max_value=60, value=30, step=1)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button(
    "Optimize stimulation pattern", type="primary", use_container_width=True
)
st.sidebar.caption(
    "Runs the full pipeline on click (a few seconds on CPU). Sliders don't "
    "auto-rerun the optimization."
)


# ==========================================================================
# Run pipeline on button click
# ==========================================================================
if run_clicked:
    if input_image is None:
        st.sidebar.error("Provide an image via upload or camera snapshot first.")
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
                lr=1e-5,
                progress_callback=_progress_cb,
            )
            progress_bar.progress(1.0, text="Optimization complete.")

            st.session_state["pv_result"] = result
            st.session_state["pv_input_image"] = input_image
            st.session_state["pv_v1_target"] = to_numpy(v1_target)
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
    v1_target_np = st.session_state["pv_v1_target"]

    final_phosphene = np.asarray(result.get("final_phosphene"))
    frames = result.get("frames") or []
    loss_trace = result.get("loss_trace") or []

    st.markdown("## Pipeline output")
    st.caption(
        f"Subject {meta['subject']} · {meta['model_variant']} · "
        f"{meta['n_electrodes_side']}×{meta['n_electrodes_side']} electrode grid "
        f"(dropout {meta['dropout']:.2f}, jitter {meta['jitter']:.2f}) · "
        f"{meta['n_steps']} optimization steps · {meta['n_v1_voxels']} V1 voxels"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
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
        with st.container(border=True):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">2</span>V1 target response</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">Predicted V1 activity, reshaped into a '
                "compact heatmap strip (illustrative, see caveats above).</div>",
                unsafe_allow_html=True,
            )
            st.image(render_vector_heatmap(v1_target_np), use_container_width=True)

    with col3:
        with st.container(border=True):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">3</span>Simulated percept</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">Final phosphene pattern from the dynaphos '
                "simulator after optimization.</div>",
                unsafe_allow_html=True,
            )
            st.image(array_to_gray_image(final_phosphene), use_container_width=True, clamp=True)

    with col4:
        with st.container(border=True):
            st.markdown(
                '<div class="pv-panel-title"><span class="pv-badge">4</span>Optimization scrubber</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pv-panel-caption">Step through intermediate snapshots to see the '
                "percept sharpen (not live video — precomputed frames).</div>",
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
                    array_to_gray_image(np.asarray(frames[frame_idx])),
                    use_container_width=True,
                    clamp=True,
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
        "Provide a photo in the sidebar (upload or camera snapshot), choose a subject "
        "and model variant, then click **Optimize stimulation pattern** to run the "
        "pipeline. It takes a few seconds on CPU — this is not a live video feed."
    )
