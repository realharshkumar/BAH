import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime, timedelta

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
    ('Hub-Core', 'Launch-Site-A',   'MPLS', 0.07),
    ('Hub-Core', 'Tracking-North',  'MPLS', 0.07),
    ('Hub-Core', 'Tracking-South',  'MPLS', 0.06),
    ('Hub-Core', 'RnD-Center',      'MPLS', 0.06),
    ('Hub-Core', 'HQ-Admin',        'MPLS', 0.08),
    ('Launch-Site-A',  'Tracking-North', 'MPLS', 0.03),
    ('Tracking-South', 'RnD-Center',     'MPLS', 0.02),
    ('Hub-Core',    'Overseas-East',   'SDWAN', 0.17),
    ('Hub-Core',    'Overseas-West',   'SDWAN', 0.18),
    ('Hub-Core',    'Overseas-Island', 'SDWAN', 0.20),
]
for src, dst, ltype, fprob in edges:
    G.add_edge(src, dst, link_type=ltype, failure_prob=fprob)

HUB_FAILURE_PROB = 0.09

def mpls_normal():
    return {
        'RSVP_TE_Latency_ms':      random.randint(10, 40),
        'OSPF_Cost_Metric':        random.choice([10, 20]),
        'Jitter_ms':               None,
        'WAN_Path_Score':          None,
        'BGP_Peer_Flaps':          random.randint(0, 1),
        'IPSec_Tunnel_Status':     1,
        'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.3), 2),
    }

def mpls_stressed():
    return {
        'RSVP_TE_Latency_ms':      random.randint(150, 450),
        'OSPF_Cost_Metric':        random.choice([80, 100, 120]),
        'Jitter_ms':               None,
        'WAN_Path_Score':          None,
        'BGP_Peer_Flaps':          random.randint(4, 12),
        'IPSec_Tunnel_Status':     random.choice([0, 1]),
        'Interface_Drop_Rate_Pct': round(random.uniform(5.0, 20.0), 2),
    }

def sdwan_normal():
    return {
        'RSVP_TE_Latency_ms':      None,
        'OSPF_Cost_Metric':        None,
        'Jitter_ms':               round(random.uniform(1.0, 10.0), 2),
        'WAN_Path_Score':          round(random.uniform(80.0, 100.0), 2),
        'BGP_Peer_Flaps':          random.randint(0, 1),
        'IPSec_Tunnel_Status':     1,
        'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.5), 2),
    }

def sdwan_stressed():
    return {
        'RSVP_TE_Latency_ms':      None,
        'OSPF_Cost_Metric':        None,
        'Jitter_ms':               round(random.uniform(40.0, 150.0), 2),
        'WAN_Path_Score':          round(random.uniform(10.0, 40.0), 2),
        'BGP_Peer_Flaps':          random.randint(3, 10),
        'IPSec_Tunnel_Status':     random.choice([0, 1]),
        'Interface_Drop_Rate_Pct': round(random.uniform(5.0, 25.0), 2),
    }

RAMP_STEPS = 4  # rows before failure that show gradual degradation

def mpls_ramp(step):
    # step: 1..RAMP_STEPS, 1 = furthest from failure, RAMP_STEPS = right before failure
    frac = step / RAMP_STEPS
    return {
        'RSVP_TE_Latency_ms':      round(random.randint(10, 40) + frac * random.randint(100, 400)),
        'OSPF_Cost_Metric':        random.choice([10, 20]) + int(frac * random.choice([60, 90, 100])),
        'Jitter_ms':               None,
        'WAN_Path_Score':          None,
        'BGP_Peer_Flaps':          int(frac * random.randint(3, 10)),
        'IPSec_Tunnel_Status':     1 if frac < 0.75 else random.choice([0, 1]),
        'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.3) + frac * random.uniform(4.0, 18.0), 2),
    }

def sdwan_ramp(step):
    frac = step / RAMP_STEPS
    return {
        'RSVP_TE_Latency_ms':      None,
        'OSPF_Cost_Metric':        None,
        'Jitter_ms':               round(random.uniform(1.0, 10.0) + frac * random.uniform(35.0, 140.0), 2),
        'WAN_Path_Score':          round(max(5.0, random.uniform(80.0, 100.0) - frac * random.uniform(50.0, 85.0)), 2),
        'BGP_Peer_Flaps':          int(frac * random.randint(3, 9)),
        'IPSec_Tunnel_Status':     1 if frac < 0.75 else random.choice([0, 1]),
        'Interface_Drop_Rate_Pct': round(random.uniform(0.0, 0.5) + frac * random.uniform(4.0, 22.0), 2),
    }

num_timesteps = 2000
start_time = datetime.now()
data = []

def schedule_next_failure(fprob):
    base_gap = int(1 / max(fprob, 0.01))
    avg_gap = max(base_gap, 36)
    return random.randint(avg_gap, int(avg_gap * 1.5))

link_countdown = {}
for src, dst, edge_attr in G.edges(data=True):
    link_countdown[(src, dst)] = schedule_next_failure(edge_attr['failure_prob'])

timestep_counter = {(s, d): 0 for s, d, _ in G.edges(data=True)}

print("Generating telemetry...")
for i in range(num_timesteps):
    timestamp = start_time + timedelta(seconds=i * 30)
    hub_failing = random.random() < HUB_FAILURE_PROB

    for src, dst, edge_attr in G.edges(data=True):
        ltype = edge_attr['link_type']
        fprob = edge_attr['failure_prob']
        key = (src, dst)
        timestep_counter[key] += 1
        link_countdown[key] -= 1

        if link_countdown[key] <= 0:
            metrics = mpls_stressed() if ltype == 'MPLS' else sdwan_stressed()
            metrics['Link_Failed'] = 1
            link_countdown[key] = schedule_next_failure(fprob)
            timestep_counter[key] = 0
        elif 0 < link_countdown[key] <= RAMP_STEPS:
            step = RAMP_STEPS - link_countdown[key] + 1  # 1..RAMP_STEPS, closer to failure = higher
            metrics = mpls_ramp(step) if ltype == 'MPLS' else sdwan_ramp(step)
            metrics['Link_Failed'] = 0
        else:
            if hub_failing:
                metrics = mpls_stressed() if ltype == 'MPLS' else sdwan_stressed()
            else:
                metrics = mpls_normal() if ltype == 'MPLS' else sdwan_normal()
            metrics['Link_Failed'] = 0

        data.append({
            'Timestamp':   timestamp,
            'Source_Node': src,
            'Dest_Node':   dst,
            'Link_Type':   ltype,
            **metrics
        })

columns = [
    'Timestamp', 'Source_Node', 'Dest_Node', 'Link_Type',
    'RSVP_TE_Latency_ms', 'OSPF_Cost_Metric',
    'Jitter_ms', 'WAN_Path_Score',
    'BGP_Peer_Flaps', 'IPSec_Tunnel_Status',
    'Interface_Drop_Rate_Pct', 'Link_Failed'
]

import os
os.makedirs('info', exist_ok=True)
df = pd.DataFrame(data, columns=columns)
df.to_csv('info/network_data.csv', index=False)
print(f"Done. {len(df)} rows, failure rate: {df['Link_Failed'].mean():.2%}")

# Topology visualization
node_colors = []
for n, attr in G.nodes(data=True):
    if attr['type'] == 'hub': node_colors.append('#e74c3c')
    elif attr['type'] == 'domestic': node_colors.append('#3498db')
    else: node_colors.append('#2ecc71')

mpls_edges  = [(u,v) for u,v in G.edges() if G[u][v]['link_type']=='MPLS']
sdwan_edges = [(u,v) for u,v in G.edges() if G[u][v]['link_type']=='SDWAN']
pos = nx.spring_layout(G, seed=42, k=2.5)
plt.figure(figsize=(16, 10))
plt.title("Network Topology", fontsize=14, fontweight='bold')
nx.draw_networkx_edges(G, pos, edgelist=mpls_edges,  edge_color='#e67e22', style='solid',  width=2)
nx.draw_networkx_edges(G, pos, edgelist=sdwan_edges, edge_color='#9b59b6', style='dashed', width=2)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')
legend_elements = [
    Patch(facecolor='#e74c3c', label='Hub'),
    Patch(facecolor='#3498db', label='Domestic'),
    Patch(facecolor='#2ecc71', label='Overseas'),
    Line2D([0],[0], color='#e67e22', linewidth=2, linestyle='solid',  label='MPLS'),
    Line2D([0],[0], color='#9b59b6', linewidth=2, linestyle='dashed', label='SD-WAN'),
]
plt.legend(handles=legend_elements, loc='lower left')
plt.axis('off')
plt.tight_layout()
plt.savefig('info/network_topology.png', dpi=150)
plt.show()
print("Topology saved to info/network_topology.png")
