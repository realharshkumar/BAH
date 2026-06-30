import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 1. BUILD NETWORK TOPOLOGY
# ─────────────────────────────────────────────
G = nx.Graph()

# Nodes: fictional space agency sites
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

# Edges: (src, dst, link_type, failure_prob)
edges = [
    # MPLS core links (hub-facing, medium failure prob)
    ('Hub-Core', 'Launch-Site-A',   'MPLS', 0.07),
    ('Hub-Core', 'Tracking-North',  'MPLS', 0.07),
    ('Hub-Core', 'Tracking-South',  'MPLS', 0.06),
    ('Hub-Core', 'RnD-Center',      'MPLS', 0.06),
    ('Hub-Core', 'HQ-Admin',        'MPLS', 0.08),
    # Domestic PE-to-PE MPLS (low failure prob)
    ('Launch-Site-A',  'Tracking-North', 'MPLS', 0.03),
    ('Tracking-South', 'RnD-Center',     'MPLS', 0.02),
    # SD-WAN overseas (high failure prob)
    ('Hub-Core',    'Overseas-East',   'SDWAN', 0.17),
    ('Hub-Core',    'Overseas-West',   'SDWAN', 0.18),
    ('Hub-Core',    'Overseas-Island', 'SDWAN', 0.20),
]

for src, dst, ltype, fprob in edges:
    G.add_edge(src, dst, link_type=ltype, failure_prob=fprob)

# Hub self-failure probability
HUB_FAILURE_PROB = 0.09

# ─────────────────────────────────────────────
# 2. TELEMETRY GENERATION HELPERS
# ─────────────────────────────────────────────

def mpls_normal():
    return {
        'RSVP_TE_Latency_ms':    random.randint(10, 40),
        'OSPF_Cost_Metric':      random.choice([10, 20]),
        'Jitter_ms':             None,
        'WAN_Path_Score':        None,
        'BGP_Peer_Flaps':        random.randint(0, 1),
        'IPSec_Tunnel_Status':   1,
        'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.3), 2),
    }

def mpls_stressed():
    return {
        'RSVP_TE_Latency_ms':    random.randint(150, 450),
        'OSPF_Cost_Metric':      random.choice([80, 100, 120]),
        'Jitter_ms':             None,
        'WAN_Path_Score':        None,
        'BGP_Peer_Flaps':        random.randint(4, 12),
        'IPSec_Tunnel_Status':   random.choice([0, 1]),
        'Interface_Drop_Rate_Pct': round(random.uniform(5.0, 20.0), 2),
    }

def mpls_precursor(severity):
    # severity: 0.2 -> 1.0, gradual degradation leading up to failure
    return {
        'RSVP_TE_Latency_ms':    int(40 + severity * 350),
        'OSPF_Cost_Metric':      random.choice([20, 40, 60, 80]),
        'Jitter_ms':             None,
        'WAN_Path_Score':        None,
        'BGP_Peer_Flaps':        random.randint(0, int(severity * 6)),
        'IPSec_Tunnel_Status':   1,
        'Interface_Drop_Rate_Pct': round(0.3 + severity * 4.0, 2),
    }

def sdwan_normal():
    return {
        'RSVP_TE_Latency_ms':    None,
        'OSPF_Cost_Metric':      None,
        'Jitter_ms':             round(random.uniform(1.0, 10.0), 2),
        'WAN_Path_Score':        round(random.uniform(80.0, 100.0), 2),
        'BGP_Peer_Flaps':        random.randint(0, 1),
        'IPSec_Tunnel_Status':   1,
        'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.5), 2),
    }

def sdwan_stressed():
    return {
        'RSVP_TE_Latency_ms':    None,
        'OSPF_Cost_Metric':      None,
        'Jitter_ms':             round(random.uniform(40.0, 150.0), 2),
        'WAN_Path_Score':        round(random.uniform(10.0, 40.0), 2),
        'BGP_Peer_Flaps':        random.randint(3, 10),
        'IPSec_Tunnel_Status':   random.choice([0, 1]),
        'Interface_Drop_Rate_Pct': round(random.uniform(5.0, 25.0), 2),
    }

def sdwan_precursor(severity):
    return {
        'RSVP_TE_Latency_ms':    None,
        'OSPF_Cost_Metric':      None,
        'Jitter_ms':             round(10.0 + severity * 130.0, 2),
        'WAN_Path_Score':        round(100.0 - severity * 65.0, 2),
        'BGP_Peer_Flaps':        random.randint(0, int(severity * 5)),
        'IPSec_Tunnel_Status':   1,
        'Interface_Drop_Rate_Pct': round(0.5 + severity * 4.0, 2),
    }

# Failure label derived from packet loss threshold
FAILURE_THRESHOLD = 5.0

def failure_label(drop_rate):
    return 1 if drop_rate >= FAILURE_THRESHOLD else 0

# Precursor window: how many timesteps BEFORE an actual failure we
# start showing gradual degradation signals. This is what makes the
# model PREDICTIVE instead of reactive.
PRECURSOR_WINDOW = 5

# ─────────────────────────────────────────────
# 3. GENERATE ROWS
# ─────────────────────────────────────────────
num_timesteps = 600  # ~600 minutes of telemetry
start_time = datetime.now()
data = []

print("Generating network telemetry...")

# Each link gets its own countdown to its next scheduled failure event.
# When countdown hits 0 -> failure row. The PRECURSOR_WINDOW rows before
# that show gradually worsening metrics (this is what lets the ML model
# predict failures ahead of time instead of just detecting them as they happen).
def schedule_next_failure(fprob):
    # Lower fprob -> failures scheduled further apart, higher fprob -> sooner.
    # Multiply by ~8x window so links spend most of their life healthy,
    # with only a brief precursor+failure blip before recovering.
    base_gap = int((1 / max(fprob, 0.01)))
    avg_gap = max(base_gap, (PRECURSOR_WINDOW + 1) * 6)
    return random.randint(avg_gap, int(avg_gap * 1.5))

link_countdown = {}
for src, dst, edge_attr in G.edges(data=True):
    link_countdown[(src, dst)] = schedule_next_failure(edge_attr['failure_prob'])

for i in range(num_timesteps):
    timestamp = start_time + timedelta(minutes=i)

    # Hub self-failure event still cascades extra stress network-wide
    hub_failing = random.random() < HUB_FAILURE_PROB

    for src, dst, edge_attr in G.edges(data=True):
        ltype = edge_attr['link_type']
        fprob = edge_attr['failure_prob']
        key = (src, dst)

        link_countdown[key] -= 1
        countdown = link_countdown[key]

        is_precursor = 0

        if countdown <= 0:
            # Failure row
            metrics = mpls_stressed() if ltype == 'MPLS' else sdwan_stressed()
            link_countdown[key] = schedule_next_failure(fprob)

        elif countdown <= PRECURSOR_WINDOW:
            # Precursor row — gradual degradation, severity rises as countdown shrinks
            severity = (PRECURSOR_WINDOW - countdown + 1) / PRECURSOR_WINDOW  # 0.2 -> 1.0
            metrics = mpls_precursor(severity) if ltype == 'MPLS' else sdwan_precursor(severity)
            is_precursor = 1

        else:
            # Normal healthy conditions, with hub-wide cascading stress as an exception
            if hub_failing:
                metrics = mpls_stressed() if ltype == 'MPLS' else sdwan_stressed()
            else:
                metrics = mpls_normal() if ltype == 'MPLS' else sdwan_normal()

        # Precursor rows are labeled as failure-imminent (predictive label),
        # not just rows that have already crossed the drop-rate threshold.
        metrics['Link_Failure_Imminent'] = 1 if is_precursor else failure_label(metrics['Interface_Drop_Rate_Pct'])
        metrics['Is_Precursor'] = is_precursor

        row = {
            'Timestamp':   timestamp,
            'Source_Node': src,
            'Dest_Node':   dst,
            'Link_Type':   ltype,
            **metrics
        }
        data.append(row)

# ─────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────
columns = [
    'Timestamp', 'Source_Node', 'Dest_Node', 'Link_Type',
    'RSVP_TE_Latency_ms', 'OSPF_Cost_Metric',
    'Jitter_ms', 'WAN_Path_Score',
    'BGP_Peer_Flaps', 'IPSec_Tunnel_Status',
    'Interface_Drop_Rate_Pct', 'Is_Precursor', 'Link_Failure_Imminent'
]

df = pd.DataFrame(data, columns=columns)
import os
os.makedirs('info', exist_ok=True)

df.to_csv('info/network_data.csv', index=False)

print(f"Done. {len(df)} rows written to 'info/network_data.csv'")
print(f"Failure-imminent rate: {df['Link_Failure_Imminent'].mean():.2%}")
print(f"Precursor rows: {df['Is_Precursor'].sum()} ({df['Is_Precursor'].mean():.2%})")

# ─────────────────────────────────────────────
# 5. VISUALIZE TOPOLOGY
# ─────────────────────────────────────────────
node_colors = []
for n, attr in G.nodes(data=True):
    if attr['type'] == 'hub':
        node_colors.append('#e74c3c')
    elif attr['type'] == 'domestic':
        node_colors.append('#3498db')
    else:
        node_colors.append('#2ecc71')

mpls_edges  = [(u, v) for u, v in G.edges() if G[u][v]['link_type'] == 'MPLS']
sdwan_edges = [(u, v) for u, v in G.edges() if G[u][v]['link_type'] == 'SDWAN']

pos = nx.spring_layout(G, seed=42, k=2.5)
plt.figure(figsize=(16, 10))
plt.title("Network Topology", fontsize=14, fontweight='bold')

nx.draw_networkx_edges(G, pos, edgelist=mpls_edges,  edge_color='#e67e22', style='solid', width=2)
nx.draw_networkx_edges(G, pos, edgelist=sdwan_edges, edge_color='#9b59b6', style='dashed', width=2)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', font_weight='bold')

legend_elements = [
    Patch(facecolor='#e74c3c', label='Hub'),
    Patch(facecolor='#3498db', label='Domestic'),
    Patch(facecolor='#2ecc71', label='Overseas'),
    Line2D([0], [0], color='#e67e22', linewidth=2, linestyle='solid',  label='MPLS'),
    Line2D([0], [0], color='#9b59b6', linewidth=2, linestyle='dashed', label='SD-WAN'),
]
plt.legend(handles=legend_elements, loc='lower left')
plt.axis('off')
plt.tight_layout()
plt.savefig('info/network_topology.png', dpi=150)
plt.show()
print("Topology saved to info/network_topology.png")