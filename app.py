import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import joblib
import pandas as pd
import random
import time
from tensorflow import keras
import numpy as np

from copilot import generate_remediation, identify_top_feature

st.set_page_config(page_title="Air-Gapped Predictive NOC Copilot", layout="wide")
st.title("Air-Gapped Predictive Network Copilot")
st.caption("Fully offline ML + LLM monitoring for MPLS / SD-WAN infrastructure")

# ─────────────────────────────────────────────
# 1. TOPOLOGY (same as data_generator.py)
# ─────────────────────────────────────────────
@st.cache_resource
def build_topology():
    G = nx.Graph()
    nodes = {
        'Hub-Core':         {'type': 'hub'},
        'Launch-Site-A':    {'type': 'domestic'},
        'Tracking-North':   {'type': 'domestic'},
        'Tracking-South':   {'type': 'domestic'},
        'RnD-Center':       {'type': 'domestic'},
        'HQ-Admin':         {'type': 'domestic'},
        'Overseas-East':    {'type': 'overseas'},
        'Overseas-West':    {'type': 'overseas'},
        'Overseas-Island':  {'type': 'overseas'},
    }
    G.add_nodes_from([(n, attr) for n, attr in nodes.items()])

    edges = [
        ('Hub-Core', 'Launch-Site-A',   'MPLS'),
        ('Hub-Core', 'Tracking-North',  'MPLS'),
        ('Hub-Core', 'Tracking-South',  'MPLS'),
        ('Hub-Core', 'RnD-Center',      'MPLS'),
        ('Hub-Core', 'HQ-Admin',        'MPLS'),
        ('Launch-Site-A',  'Tracking-North', 'MPLS'),
        ('Tracking-South', 'RnD-Center',     'MPLS'),
        ('Hub-Core',    'Overseas-East',   'SDWAN'),
        ('Hub-Core',    'Overseas-West',   'SDWAN'),
        ('Hub-Core',    'Overseas-Island', 'SDWAN'),
    ]
    for src, dst, ltype in edges:
        G.add_edge(src, dst, link_type=ltype)
    return G

G = build_topology()
all_links = [(u, v, d['link_type']) for u, v, d in G.edges(data=True)]

# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────
WINDOW_SIZE = 5

@st.cache_resource
def load_model():
    model = keras.models.load_model('network_model.keras')
    feature_cols = joblib.load('model_features.pkl')
    try:
        threshold = joblib.load('model_threshold.pkl')
    except FileNotFoundError:
        threshold = 0.5
    return model, feature_cols, threshold

model, feature_cols, THRESHOLD = load_model()

# ─────────────────────────────────────────────
# 3. METRIC GENERATORS (live simulation, mirrors data_generator.py)
# ─────────────────────────────────────────────
def mpls_normal():
    return {'RSVP_TE_Latency_ms': random.randint(10, 40), 'OSPF_Cost_Metric': random.choice([10, 20]),
            'Jitter_ms': None, 'WAN_Path_Score': None, 'BGP_Peer_Flaps': random.randint(0, 1),
            'IPSec_Tunnel_Status': 1, 'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.3), 2)}

def mpls_stressed():
    return {'RSVP_TE_Latency_ms': random.randint(150, 450), 'OSPF_Cost_Metric': random.choice([80, 100, 120]),
            'Jitter_ms': None, 'WAN_Path_Score': None, 'BGP_Peer_Flaps': random.randint(4, 12),
            'IPSec_Tunnel_Status': random.choice([0, 1]), 'Interface_Drop_Rate_Pct': round(random.uniform(5.0, 20.0), 2)}

def sdwan_normal():
    return {'RSVP_TE_Latency_ms': None, 'OSPF_Cost_Metric': None, 'Jitter_ms': round(random.uniform(1.0, 10.0), 2),
            'WAN_Path_Score': round(random.uniform(80.0, 100.0), 2), 'BGP_Peer_Flaps': random.randint(0, 1),
            'IPSec_Tunnel_Status': 1, 'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.5), 2)}

def sdwan_stressed():
    return {'RSVP_TE_Latency_ms': None, 'OSPF_Cost_Metric': None, 'Jitter_ms': round(random.uniform(40.0, 150.0), 2),
            'WAN_Path_Score': round(random.uniform(10.0, 40.0), 2), 'BGP_Peer_Flaps': random.randint(3, 10),
            'IPSec_Tunnel_Status': random.choice([0, 1]), 'Interface_Drop_Rate_Pct': round(random.uniform(5.0, 25.0), 2)}

def generate_live_metrics(link_type, stressed=False):
    if link_type == 'MPLS':
        return mpls_stressed() if stressed else mpls_normal()
    return sdwan_stressed() if stressed else sdwan_normal()

# ─────────────────────────────────────────────
# 4. RUN MODEL PREDICTION ON A LINK'S METRICS
# ─────────────────────────────────────────────
def metrics_to_row(link_type, metrics):
    row = {col: 0.0 for col in feature_cols}
    for col in ['RSVP_TE_Latency_ms', 'OSPF_Cost_Metric', 'Jitter_ms', 'WAN_Path_Score',
                'BGP_Peer_Flaps', 'IPSec_Tunnel_Status']:
        if col in row:
            row[col] = metrics.get(col) or 0

    type_col = f"Link_Type_{link_type}"
    if type_col in row:
        row[type_col] = 1.0

    return [row[col] for col in feature_cols]

def predict_link(history):
    # history: list of last WINDOW_SIZE rows (each a list of feature values)
    if len(history) < WINDOW_SIZE:
        return 0, 0.0
    X = np.array(history[-WINDOW_SIZE:], dtype=np.float32)[None, :, :]
    proba = float(model.predict(X, verbose=0)[0][0])
    pred = int(proba >= THRESHOLD)
    return pred, proba

# ─────────────────────────────────────────────
# 5. SIMULATE LIVE STATE FOR ALL LINKS (session state, persists across reruns)
# ─────────────────────────────────────────────
if "link_history" not in st.session_state:
    st.session_state.link_history = {f"{s}->{d}": [] for s, d, _ in all_links}

if "link_state" not in st.session_state:
    st.session_state.link_state = {}
    for src, dst, ltype in all_links:
        key = f"{src}->{dst}"
        stressed = random.random() < 0.15  # ~15% of links start in stressed state for demo visibility
        # seed history with WINDOW_SIZE rows so a prediction is available immediately
        for _ in range(WINDOW_SIZE):
            metrics = generate_live_metrics(ltype, stressed=stressed)
            st.session_state.link_history[key].append(metrics_to_row(ltype, metrics))
        pred, proba = predict_link(st.session_state.link_history[key])
        st.session_state.link_state[key] = {
            'src': src, 'dst': dst, 'link_type': ltype,
            'metrics': metrics, 'prediction': pred, 'confidence': proba
        }

def refresh_link_state():
    for src, dst, ltype in all_links:
        key = f"{src}->{dst}"
        stressed = random.random() < 0.15
        metrics = generate_live_metrics(ltype, stressed=stressed)
        st.session_state.link_history[key].append(metrics_to_row(ltype, metrics))
        st.session_state.link_history[key] = st.session_state.link_history[key][-WINDOW_SIZE:]
        pred, proba = predict_link(st.session_state.link_history[key])
        st.session_state.link_state[key] = {
            'src': src, 'dst': dst, 'link_type': ltype,
            'metrics': metrics, 'prediction': pred, 'confidence': proba
        }

# ─────────────────────────────────────────────
# 6. TOP CONTROLS
# ─────────────────────────────────────────────
col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("Refresh Live Telemetry", use_container_width=True):
        refresh_link_state()
        st.rerun()

# ─────────────────────────────────────────────
# 7. TOPOLOGY VISUALIZATION (failed links highlighted)
# ─────────────────────────────────────────────
def draw_topology():
    node_colors = []
    for n, attr in G.nodes(data=True):
        if attr['type'] == 'hub':
            node_colors.append('#e74c3c')
        elif attr['type'] == 'domestic':
            node_colors.append('#3498db')
        else:
            node_colors.append('#2ecc71')

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        key = f"{u}->{v}"
        state = st.session_state.link_state.get(key)
        if state and state['prediction'] == 1:
            edge_colors.append('#e74c3c')  # red = failure imminent
            edge_widths.append(3.5)
        else:
            edge_colors.append('#95a5a6')  # gray = healthy
            edge_widths.append(1.5)

    pos = nx.spring_layout(G, seed=42, k=2.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)

    legend_elements = [
        Patch(facecolor='#e74c3c', label='Hub'),
        Patch(facecolor='#3498db', label='Domestic'),
        Patch(facecolor='#2ecc71', label='Overseas'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='Failure Imminent'),
        Line2D([0], [0], color='#95a5a6', linewidth=1.5, label='Healthy'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    ax.axis('off')
    return fig

st.subheader("Network Topology — Live Status")
st.pyplot(draw_topology())

# ─────────────────────────────────────────────
# 8. FLAGGED LINKS SUMMARY
# ─────────────────────────────────────────────
flagged = [s for s in st.session_state.link_state.values() if s['prediction'] == 1]

st.subheader(f"Flagged Links ({len(flagged)})")
if flagged:
    flagged_df = pd.DataFrame([
        {
            'Link': f"{s['src']} -> {s['dst']}",
            'Type': s['link_type'],
            'Confidence': f"{s['confidence']:.0%}",
        }
        for s in flagged
    ])
    st.dataframe(flagged_df, use_container_width=True, hide_index=True)
else:
    st.success("All links healthy. No failures predicted.")

# ─────────────────────────────────────────────
# 9. LINK DROPDOWN — DRILL INTO ANY SPECIFIC LINK
# ─────────────────────────────────────────────
st.subheader("Inspect a Link")
link_keys = list(st.session_state.link_state.keys())
selected_key = st.selectbox("Select a link to inspect", link_keys)

state = st.session_state.link_state[selected_key]
metrics = state['metrics']

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"**Link:** {state['src']} → {state['dst']}")
    st.markdown(f"**Type:** {state['link_type']}")
    metric_display = {k: v for k, v in metrics.items() if v is not None}
    st.json(metric_display)

with col2:
    if state['prediction'] == 1:
        st.error(f"FAILURE IMMINENT — Confidence: {state['confidence']:.0%}")
        top_feature = identify_top_feature(state['link_type'], metrics)
        st.markdown(f"**Primary anomaly:** `{top_feature}`")

        with st.spinner("AI Copilot diagnosing..."):
            try:
                remediation = generate_remediation(
                    src_node=state['src'],
                    dst_node=state['dst'],
                    link_type=state['link_type'],
                    metrics=metrics,
                    prediction_confidence=state['confidence']
                )
                st.markdown("### AI Copilot Remediation Plan")
                st.markdown(remediation)
            except Exception as e:
                st.warning(f"Copilot unavailable (is Ollama running?): {e}")
    else:
        st.success(f"Healthy — Confidence: {(1 - state['confidence']):.0%}")
