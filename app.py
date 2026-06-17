import streamlit as st
import joblib
import ollama
import random
import time
import pandas as pd

st.set_page_config(page_title="Secure MPLS Copilot", layout="centered")
st.title("📡 Air-Gapped Predictive Network Copilot")
st.write("🔄 **Live Monitoring Active** — Ingesting network telemetry continuously...")

# 1. Load your trained machine learning model
model = joblib.load('network_model.pkl')

# 2. Initialize a session state list to store historical latency for the graph
if "latency_history" not in st.session_state:
    st.session_state.latency_history = [20] * 10  # Pre-fill with normal data

# 3. Create UI containers
metrics_box = st.empty()
chart_box = st.empty()
alert_box = st.empty()
copilot_box = st.empty()

# 4. Infinite loop to simulate real-time data streaming
while True:
    # 10% chance to drop into a failure state for simulation
    if random.random() > 0.9: 
        latency, packet_loss, bandwidth = random.randint(180, 350), round(random.uniform(6.0, 15.0), 2), random.randint(85, 98)
    else:
        latency, packet_loss, bandwidth = random.randint(12, 28), round(random.uniform(0.1, 0.4), 2), random.randint(35, 55)
        
    # Update latency history (keep last 10 points)
    st.session_state.latency_history.append(latency)
    st.session_state.latency_history.pop(0)
    
    # Overwrite the metrics box
    metrics_box.info(f"📊 **Live Metrics** | Latency: {latency}ms | Packet Loss: {packet_loss}% | Bandwidth: {bandwidth}%")
    
    # Overwrite the chart box with an updated line graph
    chart_box.line_chart(pd.DataFrame(st.session_state.latency_history, columns=["Latency (ms)"]))
    
    # 5. Machine learning model predicts based on the fresh data
    prediction = model.predict([[latency, packet_loss, bandwidth]])
    
    if prediction[0] == 1:
        alert_box.error("🚨 CRITICAL ALERT: Imminent MPLS Link Failure Detected!")
        copilot_box.markdown("⏳ *AI Copilot is diagnosing local topology...*")
        
        prompt = f"The predictive model flagged a network crash. Stats: Latency {latency}ms, Packet Loss {packet_loss}%, Bandwidth {bandwidth}%. Give me 3 short bullet points to fix this."
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        
        copilot_box.markdown(f"### 🤖 AI Copilot Remediation Plan\n{response['message']['content']}")
    else:
        alert_box.success("✅ Network Status: HEALTHY. No anomalies detected.")
        copilot_box.empty()
        
    time.sleep(3)  # Speed up loop to 3 seconds for a snappier presentation demo