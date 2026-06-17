import pandas as pd
import random
from datetime import datetime, timedelta

# Configuration
num_rows = 1000
data = []
start_time = datetime.now()

for i in range(num_rows):
    timestamp = start_time + timedelta(minutes=i)
    router_id = f"Router_{random.randint(1, 3)}"
    
    # Normal network conditions
    latency = random.randint(10, 30)
    packet_loss = round(random.uniform(0.0, 0.5), 2)
    bandwidth = random.randint(30, 60)
    link_failure = 0
    
    # Inject a failure pattern every 50 rows
    if i % random.randint(48,50) == 0:  
        latency = random.randint(150, 400)      # Spike latency
        packet_loss = round(random.uniform(5.0, 20.0), 2)  # Spike packet loss
        bandwidth = random.randint(90, 100)     # Max out bandwidth
        link_failure = 1                         # The network crashes

    data.append([timestamp, router_id, latency, packet_loss, bandwidth, link_failure])

# Save to CSV
columns = ['Timestamp', 'Router_ID', 'Latency_ms', 'Packet_Loss_Pct', 'Bandwidth_Usage_Pct', 'Link_Failure']
df = pd.DataFrame(data, columns=columns)
df.to_csv('network_data.csv', index=False)

print("Success: network_data.csv created with 1,000 rows!")