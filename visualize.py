import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('info/network_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

LINK_SRC = 'Hub-Core'
LINK_DST = 'Launch-Site-A'
data = df[(df['Source_Node']==LINK_SRC) & (df['Dest_Node']==LINK_DST)].reset_index(drop=True)

numeric_cols = ['RSVP_TE_Latency_ms', 'OSPF_Cost_Metric', 'Jitter_ms', 'WAN_Path_Score',
                'BGP_Peer_Flaps', 'IPSec_Tunnel_Status', 'Interface_Drop_Rate_Pct',
                'Is_Precursor', 'Link_Failure_Imminent']
plot_cols = [c for c in numeric_cols if c in data.columns and data[c].notna().any()]

fig, axes = plt.subplots(len(plot_cols), 1, figsize=(16, 2.5 * len(plot_cols)), sharex=True)
fig.suptitle(f'Telemetry: {LINK_SRC} -> {LINK_DST}', fontsize=14, fontweight='bold')

for ax, col in zip(axes, plot_cols):
    ax.plot(data['Timestamp'], data[col], linewidth=0.8, color='steelblue')
    ax.fill_between(data['Timestamp'], data[col].min(), data[col].max(),
                    where=data['Link_Failure_Imminent']==1, alpha=0.15, color='red', label='Failure imminent')
    ax.set_ylabel(col, fontsize=8)
    ax.grid(True, alpha=0.3)

axes[0].legend(loc='upper right', fontsize=8)
plt.xlabel('Timestamp')
plt.tight_layout()
plt.savefig('info/telemetry_plot.png', dpi=150)
print("Saved to info/telemetry_plot.png")
