import sys
import os
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DMS Functions Explorer · Digit Recognition AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS — match the premium light theme exactly
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:            #f1f5f9;
    --surface:       #ffffff;
    --surface-2:     #f8fafc;
    --border:        #e2e8f0;
    --blue:          #3b82f6;
    --blue-dark:     #1d4ed8;
    --blue-light:    #eff6ff;
    --blue-glow:     rgba(59,130,246,0.15);
    --purple:        #8b5cf6;
    --purple-dark:   #6d28d9;
    --purple-light:  #f5f3ff;
    --green:         #10b981;
    --green-light:   #d1fae5;
    --text:          #0f172a;
    --text-2:        #475569;
    --text-3:        #94a3b8;
    --radius:        12px;
    --radius-lg:     18px;
    --radius-full:   9999px;
    --shadow:        0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    --shadow-md:     0 4px 8px rgba(0,0,0,0.08), 0 16px 32px rgba(0,0,0,0.06);
    --shadow-blue:   0 4px 14px rgba(59,130,246,0.35);
    --ease:          cubic-bezier(0.4,0,0.2,1);
}

/* ── GLOBAL BODY ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(59,130,246,0.07) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 10%, rgba(139,92,246,0.05) 0%, transparent 50%) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

[data-testid="stAppViewContainer"] > section > div {
    padding-top: 0 !important;
}

[data-testid="stVerticalBlock"] > div {
    gap: 0 !important;
}

/* ── HEADER ── */
.app-header {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 0;
    position: sticky;
    top: 0;
    z-index: 200;
    box-shadow: 0 1px 0 var(--border), 0 4px 16px rgba(0,0,0,0.04);
}

.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: var(--radius-full);
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(59,130,246,0.3);
    width: fit-content;
}

.header-title {
    font-size: 1.65rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #0f172a 30%, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.25rem 0;
    line-height: 1.1;
}

.header-sub {
    font-size: 0.9rem;
    color: var(--text-2);
    font-weight: 500;
    margin: 0;
}

.team-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 1.25rem;
    box-shadow: var(--shadow);
}

.team-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--blue);
    margin-bottom: 0.5rem;
}

.team-members {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    justify-content: flex-end;
}

.team-members span {
    font-size: 0.75rem;
    font-weight: 600;
    background: var(--blue-light);
    color: var(--blue-dark);
    padding: 0.2rem 0.6rem;
    border-radius: var(--radius-full);
}

/* ── CONCEPT STRIP ── */
.concept-strip {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0.85rem 2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    overflow-x: auto;
    margin-bottom: 1.5rem;
    flex-wrap: nowrap;
}

.concept-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.4rem 1rem;
    border-radius: var(--radius-full);
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text-2);
    white-space: nowrap;
    cursor: default;
}

.concept-chip.active {
    background: var(--blue);
    border-color: var(--blue);
    color: #fff;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35);
}

.chip-num {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(255,255,255,0.25);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 800;
}

.chip-divider { color: var(--text-3); }

/* ── CARD ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    overflow: hidden;
    height: 100%;
    transition: box-shadow 0.2s var(--ease);
}

.card:hover { box-shadow: var(--shadow-md); }

.card-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.25rem 1.5rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
}

.card-icon { font-size: 1.5rem; line-height: 1; flex-shrink: 0; }

.card-header h2 {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0 0 0.1rem 0;
}

.card-sub {
    font-size: 0.78rem;
    color: var(--purple);
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: 0;
}

.card-body { padding: 1.5rem; }

/* ── CALLOUT ── */
.concept-callout {
    background: var(--blue-light);
    border-left: 4px solid var(--blue);
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--text-2);
    margin-bottom: 1.25rem;
    line-height: 1.6;
}

.concept-callout.purple {
    background: var(--purple-light);
    border-left-color: var(--purple);
}

.concept-callout strong { color: var(--text); font-weight: 700; }

/* ── RESULT BOX ── */
.result-row {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    background: linear-gradient(135deg, #eff6ff, #f5f3ff);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

.result-label {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-3);
    margin-bottom: 0.2rem;
}

.result-digit {
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.05em;
}

.result-status { font-size: 0.85rem; font-weight: 600; color: var(--text-2); margin-bottom: 0.25rem; }
.result-conf { font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: var(--green); font-weight: 700; }

/* ── PREVIEW ROW ── */
.preview-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin-top: 1rem;
}

.preview-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-3);
    margin-bottom: 0.35rem;
    text-align: center;
}

.preview-arrow { font-size: 1.5rem; color: var(--text-3); }

/* ── NETWORK VIS ── */
.network-vis {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 1.25rem;
}

.net-layer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.25rem;
}

.net-layer-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}

.net-layer-label span:last-child {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-3);
    font-weight: 400;
}

.net-arrow-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.35rem 0;
    gap: 3px;
}

.net-arrow-line {
    width: 2px;
    height: 12px;
    background: linear-gradient(to bottom, #e2e8f0, #3b82f6);
    border-radius: 2px;
}

.net-arrow-pill {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--purple);
    background: var(--purple-light);
    border: 1px solid rgba(139,92,246,0.2);
    padding: 0.15rem 0.65rem;
    border-radius: var(--radius-full);
}

/* ── ANALYSIS BLOCK ── */
.analysis-block {
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.analysis-block:last-child {
    border-bottom: none;
    padding-bottom: 0;
    margin-bottom: 0;
}

.analysis-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.ab-left { display: flex; align-items: center; gap: 0.85rem; }

.analysis-num {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--blue);
    color: #fff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 900;
    box-shadow: 0 2px 8px rgba(59,130,246,0.3);
    flex-shrink: 0;
}

.analysis-num.purple {
    background: var(--purple);
    box-shadow: 0 2px 8px rgba(139,92,246,0.3);
}

.ab-title { font-size: 0.95rem; font-weight: 800; color: var(--text); margin: 0; }
.ab-sub { font-size: 0.72rem; color: var(--text-3); font-weight: 600; font-family: 'JetBrains Mono', monospace; margin: 0; }

.status-pill {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.3rem 0.8rem;
    border-radius: var(--radius-full);
    background: var(--green-light);
    color: var(--green);
    white-space: nowrap;
}

.status-pill.purple { background: var(--purple-light); color: var(--purple); }

/* ── INFO CARD ── */
.info-card {
    background: linear-gradient(135deg, var(--blue-light), var(--purple-light));
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
}

.info-card-title { font-size: 0.85rem; font-weight: 800; color: var(--text); margin: 0 0 0.5rem 0; }
.info-card p { font-size: 0.82rem; color: var(--text-2); line-height: 1.65; margin: 0; }

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
    font-size: 0.9rem !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    color: white !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
}

[data-testid="stButton"] > button[kind="secondary"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
}

[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
}

/* Remove all default column paddings to allow custom spacing */
[data-testid="stHorizontalBlock"] {
    gap: 1.5rem !important;
    padding: 0 1.5rem !important;
}

[data-testid="column"] {
    padding: 0 !important;
    min-width: 0 !important;   /* prevent column blowout on small screens */
}

/* Canvas iframe — never overflow its column, no clipping */
[data-testid="column"] iframe {
    max-width: 100% !important;
    display: block !important;
    border-radius: 10px;
}

/* ── RESPONSIVE ── */
@media (max-width: 768px) {
    .app-header { padding: 1rem; flex-direction: column; align-items: flex-start; }
    .header-title { font-size: 1.2rem; }
    .team-members { justify-content: flex-start; }
    [data-testid="stHorizontalBlock"] { padding: 0 0.75rem !important; }
    .concept-strip { padding: 0.75rem 1rem; gap: 0.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# ML MODEL — cached so it only trains once
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    X_scaled = X / 16.0

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_scaled, y)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    indices = np.random.RandomState(0).choice(len(X_scaled), 300, replace=False)
    embeddings = [{'x': float(X_pca[i, 0]), 'y': float(X_pca[i, 1]), 'label': int(y[i])} for i in indices]

    mean_images = {}
    for digit in range(10):
        digit_imgs = X_scaled[y == digit]
        mean_images[str(digit)] = np.mean(digit_imgs, axis=0).tolist()

    return model, pca, embeddings, mean_images

model, pca, embeddings, mean_images = load_model()


# ──────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────
if 'prediction'    not in st.session_state: st.session_state.prediction    = None
if 'probabilities' not in st.session_state: st.session_state.probabilities = None
if 'pca_point'     not in st.session_state: st.session_state.pca_point     = None
if 'confidence'    not in st.session_state: st.session_state.confidence    = None
if 'grid_data'     not in st.session_state: st.session_state.grid_data     = None
if 'inverse_img'   not in st.session_state: st.session_state.inverse_img   = None
if 'canvas_key'    not in st.session_state: st.session_state.canvas_key    = 0


# ──────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <div class="pill-badge">VTU · DMS · Group 9 · Module 3</div>
    <h1 class="header-title">Functions in Machine Learning</h1>
    <p class="header-sub">Draw a digit → Watch the neural network classify it live</p>
  </div>
  <div class="team-card">
    <div class="team-label">Team Members</div>
    <div class="team-members">
      <span>Vignesh B S</span>
      <span>Rohit Maiya M</span>
      <span>Sanskar Patil</span>
      <span>Shailesh Nayak</span>
      <span>Shreenivas CS</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONCEPT STRIP
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="concept-strip">
  <div class="concept-chip active"><span class="chip-num">1</span> Plain Function</div>
  <span class="chip-divider">→</span>
  <div class="concept-chip"><span class="chip-num">2</span> Composition</div>
  <span class="chip-divider">→</span>
  <div class="concept-chip"><span class="chip-num">3</span> Surjective</div>
  <span class="chip-divider">→</span>
  <div class="concept-chip"><span class="chip-num">4</span> Injective</div>
  <span class="chip-divider">→</span>
  <div class="concept-chip"><span class="chip-num">5</span> Inverse</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def canvas_to_grid(image_data):
    """Convert RGBA image from st_canvas to 8x8 float grid [0,1].
    
    st_canvas with background_color='#ffffff' bakes the white background
    into the image — so alpha is 255 for ALL pixels (useless for ink detection).
    Instead, we use inverted luminance: white=0 ink, dark stroke=high ink.
    """
    if image_data is None:
        return None
    arr = np.array(image_data, dtype=float)
    if arr.ndim < 3 or arr.shape[0] == 0:
        return None

    # Use RGB luminance — white background → luminance≈255 → ink≈0
    # Dark stroke (#1e293b = 30,41,59) → luminance≈39 → ink≈0.85
    r_ch = arr[:, :, 0]
    g_ch = arr[:, :, 1]
    b_ch = arr[:, :, 2]
    luminance = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    ink = (255.0 - luminance) / 255.0   # 0=white/no ink, 1=black/full ink

    H, W = ink.shape
    grid = np.zeros(64, dtype=float)
    bh = H / 8
    bw = W / 8
    for row in range(8):
        for col in range(8):
            y0, y1 = int(row * bh), int((row + 1) * bh)
            x0, x1 = int(col * bw), int((col + 1) * bw)
            block = ink[y0:y1, x0:x1]
            # boost slightly so light strokes register better
            grid[row * 8 + col] = min(float(np.mean(block)) * 1.5, 1.0)
    return grid


def make_probability_chart(probs, prediction):
    colors = ['rgba(16,185,129,0.8)' if i == prediction else 'rgba(59,130,246,0.55)'
              for i in range(10)]
    border_colors = ['#10b981' if i == prediction else 'rgba(59,130,246,0.9)'
                     for i in range(10)]

    fig = go.Figure(go.Bar(
        x=list(range(10)),
        y=[round(p * 100, 2) for p in probs],
        marker_color=colors,
        marker_line_color=border_colors,
        marker_line_width=1.5,
        text=[f'{p*100:.1f}%' if i == prediction else '' for i, p in enumerate(probs)],
        textposition='outside',
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(248,250,252,0)',
        plot_bgcolor='rgba(248,250,252,0)',
        height=220,
        xaxis=dict(tickmode='array', tickvals=list(range(10)),
                   ticktext=[str(i) for i in range(10)],
                   tickfont=dict(size=11, family='Inter', color='#475569'),
                   showgrid=False),
        yaxis=dict(range=[0, 110], tickfont=dict(size=9, color='#94a3b8'),
                   gridcolor='rgba(0,0,0,0.04)', zeroline=False),
        showlegend=False,
        font=dict(family='Inter'),
    )
    return fig


def make_scatter_chart(embeddings, pca_point=None, prediction=None):
    palette = ['#ef4444','#f97316','#eab308','#84cc16',
               '#10b981','#06b6d4','#3b82f6','#6366f1',
               '#8b5cf6','#d946ef']

    classes = [[] for _ in range(10)]
    for pt in embeddings:
        classes[pt['label']].append({'x': pt['x'], 'y': pt['y']})

    traces = []
    for idx, pts in enumerate(classes):
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        traces.append(go.Scatter(
            x=xs, y=ys, mode='markers',
            name=str(idx),
            marker=dict(color=palette[idx], size=5, opacity=0.75),
        ))

    if pca_point is not None:
        traces.append(go.Scatter(
            x=[pca_point[0]], y=[pca_point[1]],
            mode='markers+text',
            name=f'Your Input → {prediction}',
            text=['⭐'],
            textposition='top center',
            marker=dict(color='#f43f5e', size=16, symbol='star',
                        line=dict(color='white', width=2)),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(248,250,252,0)',
        plot_bgcolor='rgba(248,250,252,0)',
        height=230,
        xaxis=dict(showticklabels=False, gridcolor='rgba(0,0,0,0.03)', zeroline=False),
        yaxis=dict(showticklabels=False, gridcolor='rgba(0,0,0,0.03)', zeroline=False),
        legend=dict(font=dict(size=9, family='Inter'), orientation='h',
                    yanchor='bottom', y=1.02, xanchor='left', x=0,
                    itemsizing='constant'),
        font=dict(family='Inter'),
    )
    return fig


def make_network_html(grid_data=None, probs=None, prediction=None):
    """Render the animated Neural Network visualization as HTML."""
    # Input nodes (64)
    input_nodes_html = ''
    for i in range(64):
        val = float(grid_data[i]) if grid_data is not None else 0.0
        alpha = max(0.05, min(0.95, 0.05 + val * 0.9))
        color = f'rgba(59,130,246,{alpha:.2f})'
        input_nodes_html += f'<div style="aspect-ratio:1;background:{color};border-radius:2px;transition:background 0.3s;"></div>'

    # H1 nodes (32) - random activation after prediction
    h1_html = ''
    import random
    rng = random.Random(42 if prediction is None else prediction)
    for i in range(32):
        if prediction is not None and rng.random() > 0.35:
            color = '#60a5fa'
            scale = 'scale(1.25)'
        else:
            color = '#e2e8f0'
            scale = 'scale(1)'
        h1_html += f'<div style="width:12px;height:12px;background:{color};border-radius:50%;transform:{scale};transition:all 0.3s;flex-shrink:0;"></div>'

    # H2 nodes (16)
    h2_html = ''
    rng2 = random.Random(99 if prediction is None else prediction + 10)
    for i in range(16):
        if prediction is not None and rng2.random() > 0.25:
            color = '#818cf8'
            scale = 'scale(1.3)'
        else:
            color = '#e2e8f0'
            scale = 'scale(1)'
        h2_html += f'<div style="width:12px;height:12px;background:{color};border-radius:50%;transform:{scale};transition:all 0.3s;flex-shrink:0;"></div>'

    # Output nodes (10)
    out_html = ''
    for i in range(10):
        if prediction is not None:
            if i == prediction:
                color = '#10b981'
                scale = 'scale(1.6)'
                shadow = '0 0 8px rgba(16,185,129,0.5)'
            else:
                p = probs[i] if probs is not None else 0
                a = min(0.7, 0.1 + p * 3)
                color = f'rgba(59,130,246,{a:.2f})'
                scale = 'scale(1)'
                shadow = 'none'
        else:
            color = '#e2e8f0'; scale = 'scale(1)'; shadow = 'none'

        out_html += f'''
        <div style="display:flex;flex-direction:column;align-items:center;gap:3px;font-size:0.7rem;font-family:JetBrains Mono,monospace;font-weight:600;color:#475569;">
          <div style="width:17px;height:17px;background:{color};border-radius:50%;transform:{scale};box-shadow:{shadow};transition:all 0.3s;"></div>
          <span>{i}</span>
        </div>'''

    return f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;margin-bottom:1rem;">
  <!-- Input Layer -->
  <div style="background:white;border:1px solid #e2e8f0;border-radius:6px;padding:0.85rem 1rem;margin-bottom:0.25rem;">
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
      <span>Input x</span><span style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-weight:400;">64 features</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(16,1fr);gap:3px;">{input_nodes_html}</div>
  </div>
  <!-- Arrow 1 -->
  <div style="display:flex;flex-direction:column;align-items:center;padding:0.35rem 0;gap:3px;">
    <div style="width:2px;height:12px;background:linear-gradient(to bottom,#e2e8f0,#3b82f6);border-radius:2px;"></div>
    <span style="font-size:0.68rem;font-family:JetBrains Mono,monospace;font-weight:600;color:#8b5cf6;background:#f5f3ff;border:1px solid rgba(139,92,246,0.2);padding:0.15rem 0.65rem;border-radius:9999px;">f₁ · ReLU · 32 units</span>
  </div>
  <!-- H1 Layer -->
  <div style="background:white;border:1px solid #e2e8f0;border-radius:6px;padding:0.85rem 1rem;margin-bottom:0.25rem;">
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
      <span>Hidden h₁</span><span style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-weight:400;">32 neurons</span>
    </div>
    <div style="display:flex;gap:5px;flex-wrap:wrap;justify-content:center;">{h1_html}</div>
  </div>
  <!-- Arrow 2 -->
  <div style="display:flex;flex-direction:column;align-items:center;padding:0.35rem 0;gap:3px;">
    <div style="width:2px;height:12px;background:linear-gradient(to bottom,#e2e8f0,#3b82f6);border-radius:2px;"></div>
    <span style="font-size:0.68rem;font-family:JetBrains Mono,monospace;font-weight:600;color:#8b5cf6;background:#f5f3ff;border:1px solid rgba(139,92,246,0.2);padding:0.15rem 0.65rem;border-radius:9999px;">f₂ · ReLU · 16 units</span>
  </div>
  <!-- H2 Layer -->
  <div style="background:white;border:1px solid #e2e8f0;border-radius:6px;padding:0.85rem 1rem;margin-bottom:0.25rem;">
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
      <span>Hidden h₂</span><span style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-weight:400;">16 neurons</span>
    </div>
    <div style="display:flex;gap:5px;flex-wrap:wrap;justify-content:center;">{h2_html}</div>
  </div>
  <!-- Arrow 3 -->
  <div style="display:flex;flex-direction:column;align-items:center;padding:0.35rem 0;gap:3px;">
    <div style="width:2px;height:12px;background:linear-gradient(to bottom,#e2e8f0,#3b82f6);border-radius:2px;"></div>
    <span style="font-size:0.68rem;font-family:JetBrains Mono,monospace;font-weight:600;color:#8b5cf6;background:#f5f3ff;border:1px solid rgba(139,92,246,0.2);padding:0.15rem 0.65rem;border-radius:9999px;">f₃ · Softmax · 10 classes</span>
  </div>
  <!-- Output Layer -->
  <div style="background:white;border:1px solid #e2e8f0;border-radius:6px;padding:0.85rem 1rem;">
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
      <span>Output ŷ</span><span style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-weight:400;">10 classes</span>
    </div>
    <div style="display:flex;justify-content:space-between;gap:2px;">{out_html}</div>
  </div>
</div>
"""


def make_pixel_grid_html(grid_data, size=80, label=""):
    """Render an 8x8 pixel grid as HTML table."""
    cells = ''
    cell_size = size // 8
    for r in range(8):
        for c in range(8):
            val = float(grid_data[r * 8 + c])
            gray = int(255 - val * 255)
            cells += f'<td style="width:{cell_size}px;height:{cell_size}px;background:rgb({gray},{gray},{gray});padding:0;border:none;"></td>'
        cells = f'<tr>{cells}</tr>'
        cells = '' if r == 7 else cells  # don't accumulate, rebuild properly
    # Rebuild properly
    rows = ''
    for r in range(8):
        row = ''
        for c in range(8):
            val = float(grid_data[r * 8 + c])
            gray = int(255 - val * 255)
            row += f'<td style="width:{cell_size}px;height:{cell_size}px;background:rgb({gray},{gray},{gray});padding:0;border:none;"></td>'
        rows += f'<tr>{row}</tr>'
    return f'''
<div style="text-align:center;">
  <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#94a3b8;margin-bottom:0.35rem;">{label}</div>
  <table style="border-collapse:collapse;border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;display:inline-block;image-rendering:pixelated;">
    <tbody>{rows}</tbody>
  </table>
</div>'''


# ──────────────────────────────────────────────────────────────────
# MAIN 3-COLUMN LAYOUT
# ──────────────────────────────────────────────────────────────────
col_draw, col_net, col_analysis = st.columns([1.1, 1, 1.4])

# ─── LEFT: DRAWING PANEL ─────────────────────────────────────────
with col_draw:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon">✍️</div>
        <div>
          <h2>Draw a Digit (0–9)</h2>
          <p class="card-sub">Domain: ℝ⁶⁴ → Codomain: {0,1,…,9}</p>
        </div>
      </div>
      <div class="card-body">
        <div class="concept-callout">
          <strong>Plain Function:</strong> Every element in the domain maps to <em>exactly one</em> element in the codomain. Your drawing → one digit prediction.
        </div>
    """, unsafe_allow_html=True)

    # Drawing canvas — key changes on clear to force remount (clears strokes)
    # Width 260 keeps it safely inside the column on all screen sizes
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=20,
        stroke_color="#1e293b",
        background_color="#ffffff",
        height=260,
        width=260,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"drawing_canvas_{st.session_state.canvas_key}",
    )

    # Buttons
    bc1, bc2 = st.columns(2)
    with bc1:
        clear = st.button("↺  Clear", key="clear_btn", use_container_width=True)
    with bc2:
        evaluate = st.button("▶  Evaluate f(x)", key="eval_btn",
                             type="primary", use_container_width=True)

    if clear:
        st.session_state.prediction    = None
        st.session_state.probabilities = None
        st.session_state.pca_point     = None
        st.session_state.confidence    = None
        st.session_state.grid_data     = None
        st.session_state.inverse_img   = None
        st.session_state.canvas_key   += 1   # ← remounts canvas, wiping all strokes
        st.rerun()

    # Process canvas on evaluate
    if evaluate:
        # ── Reliable stroke detection via json_data ──────────────────
        # image_data pixel sum can be fooled by anti-aliasing noise;
        # json_data["objects"] only contains actual user-drawn paths.
        has_strokes = (
            canvas_result is not None
            and canvas_result.json_data is not None
            and len(canvas_result.json_data.get("objects", [])) > 0
        )

        if not has_strokes:
            st.warning("✏️ Please draw a digit on the canvas first!")
        else:
            grid = canvas_to_grid(canvas_result.image_data)
            if grid is None or np.sum(grid) < 0.5:
                # Very little ink → image might not have loaded yet, try again
                st.warning("⚠️ Drawing didn't register clearly. Try drawing a bolder stroke!")
            else:
                features = grid.reshape(1, -1)
                pred = int(model.predict(features)[0])
                probs = model.predict_proba(features)[0]
                pca_pt = pca.transform(features)[0]
                conf = probs[pred] * 100
                st.session_state.prediction    = pred
                st.session_state.probabilities = probs.tolist()
                st.session_state.pca_point     = pca_pt.tolist()
                st.session_state.confidence    = conf
                st.session_state.grid_data     = grid.tolist()
                st.session_state.inverse_img   = mean_images[str(pred)]

    # Result box
    pred = st.session_state.prediction
    conf = st.session_state.confidence
    if pred is not None:
        st.markdown(f"""
        <div class="result-row">
          <div style="text-align:center;flex-shrink:0;">
            <div class="result-label">f(drawing)</div>
            <div class="result-digit">{pred}</div>
          </div>
          <div>
            <div class="result-label">Status</div>
            <div class="result-status">f(input) = {pred} · Mapping complete</div>
            <div class="result-conf">Confidence: {conf:.1f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-row">
          <div style="text-align:center;flex-shrink:0;">
            <div class="result-label">f(drawing)</div>
            <div class="result-digit" style="color:#94a3b8;-webkit-text-fill-color:#94a3b8;">?</div>
          </div>
          <div>
            <div class="result-label">Status</div>
            <div class="result-status">Draw a digit and evaluate</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Preview row (8×8 downsampled + inverse)
    if st.session_state.grid_data and st.session_state.inverse_img:
        gd = st.session_state.grid_data
        inv = st.session_state.inverse_img
        input_html  = make_pixel_grid_html(gd,  label="Your Input (8×8)")
        inverse_html = make_pixel_grid_html(inv, label="f⁻¹(y) Ideal Shape")
        st.markdown(f"""
        <div class="preview-row">
          {input_html}
          <span class="preview-arrow">→</span>
          {inverse_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-row" style="justify-content:center;color:#94a3b8;font-size:0.82rem;font-weight:600;padding:1.5rem;">
          Preview appears after evaluation
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ─── MIDDLE: NEURAL NETWORK ──────────────────────────────────────
with col_net:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon">🧠</div>
        <div>
          <h2>Neural Network Flow</h2>
          <p class="card-sub">f(x) = (f₃ ∘ f₂ ∘ f₁)(x)</p>
        </div>
      </div>
      <div class="card-body">
        <div class="concept-callout purple">
          <strong>Composition:</strong> Each layer is a function. The network chains them: Output = f₃(f₂(f₁(Input))). This is function composition.
        </div>
    """, unsafe_allow_html=True)

    gd = st.session_state.grid_data
    pr = st.session_state.probabilities
    pd_ = st.session_state.prediction

    net_html = make_network_html(
        grid_data=gd,
        probs=pr,
        prediction=pd_
    )
    st.markdown(net_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
      <p class="info-card-title">🕊️ Pigeonhole Principle</p>
      <p>We are mapping <strong>2⁶⁴ possible inputs</strong> → just <strong>10 output classes</strong>.
      Since the domain is astronomically larger than the codomain, many distinct drawings
      <em>must</em> produce the same prediction. Collisions are mathematically guaranteed.</p>
    </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── RIGHT: ANALYSIS ─────────────────────────────────────────────
with col_analysis:
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon">📊</div>
        <div>
          <h2>Function Analysis</h2>
          <p class="card-sub">Surjective · Injective · Inverse</p>
        </div>
      </div>
      <div class="card-body">
    """, unsafe_allow_html=True)

    # ── Surjective block ──
    st.markdown("""
    <div class="analysis-block">
      <div class="analysis-block-header">
        <div class="ab-left">
          <span class="analysis-num">3</span>
          <div>
            <p class="ab-title">Surjective (Onto)</p>
            <p class="ab-sub">Range = Codomain</p>
          </div>
        </div>
        <span class="status-pill">✓ Confirmed</span>
      </div>
      <div class="concept-callout" style="margin-bottom:0.75rem;">
        <strong>Surjective:</strong> Every digit in {0–9} is reachable. Softmax always assigns non-zero probability to every class.
      </div>
    </div>
    """, unsafe_allow_html=True)

    probs = st.session_state.probabilities
    pred  = st.session_state.prediction
    if probs:
        fig_prob = make_probability_chart(probs, pred)
    else:
        fig_prob = make_probability_chart([0.1]*10, None)
    st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})

    # ── Injective block ──
    st.markdown("""
    <div class="analysis-block" style="margin-top:0.5rem;">
      <div class="analysis-block-header">
        <div class="ab-left">
          <span class="analysis-num purple">4</span>
          <div>
            <p class="ab-title">Injective (1-to-1)</p>
            <p class="ab-sub">f(a) = f(b) ⟹ a = b</p>
          </div>
        </div>
        <span class="status-pill purple">Not Injective</span>
      </div>
      <div class="concept-callout purple" style="margin-bottom:0.75rem;">
        <strong>Injective:</strong> The PCA scatter shows distinct digit clusters in 2D space. Your input (⭐) is plotted after evaluation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    pca_pt = st.session_state.pca_point
    fig_scatter = make_scatter_chart(embeddings, pca_pt, pred)
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    st.markdown("</div></div>", unsafe_allow_html=True)
