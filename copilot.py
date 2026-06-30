import ollama
import re

# ─────────────────────────────────────────────
# 1. LOAD RUNBOOKS
# ─────────────────────────────────────────────
# Each runbook entry is separated by '---'. We split them into individual
# chunks so retrieval pulls back the SPECIFIC relevant entry, not the
# entire file.
#
# NOTE: We use simple keyword-overlap retrieval instead of a vector DB
# (chromadb/embeddings) because the default embedding model requires an
# internet download on first run — which defeats the purpose of a fully
# offline, air-gapped copilot. This keyword approach needs zero downloads
# and zero network access, matching the PS requirement exactly.

def load_runbook_chunks(path='runbooks/network_runbooks.txt'):
    with open(path, 'r') as f:
        content = f.read()
    chunks = [c.strip() for c in content.split('---') if c.strip()]
    return chunks

_RUNBOOK_CHUNKS = None

def get_runbooks():
    global _RUNBOOK_CHUNKS
    if _RUNBOOK_CHUNKS is None:
        _RUNBOOK_CHUNKS = load_runbook_chunks()
    return _RUNBOOK_CHUNKS

# ─────────────────────────────────────────────
# 2. RETRIEVE RELEVANT RUNBOOK CONTEXT (keyword overlap, fully offline)
# ─────────────────────────────────────────────
def retrieve_context(link_type, top_feature, n_results=2):
    chunks = get_runbooks()

    # Build search keywords from link type + the triggering feature name
    keywords = set(re.findall(r'[a-z]+', link_type.lower()))
    keywords |= set(re.findall(r'[a-z]+', top_feature.lower()))

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(kw) for kw in keywords if len(kw) > 2)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for s, c in scored[:n_results] if s > 0]

    if not top_chunks:
        # fallback: return the hub cascading runbook + first match as generic context
        top_chunks = [scored[0][1]] if scored else []

    return "\n\n".join(top_chunks)

# ─────────────────────────────────────────────
# 3. IDENTIFY THE TOP CONTRIBUTING FEATURE
# ─────────────────────────────────────────────
# Given the live metrics for a flagged link, figure out which metric is most
# abnormal relative to healthy baseline. This drives both the RAG query and
# gives the LLM a concrete "why" instead of dumping all numbers at it.

HEALTHY_BASELINES = {
    'MPLS':  {'RSVP_TE_Latency_ms': 25, 'OSPF_Cost_Metric': 15, 'BGP_Peer_Flaps': 0.5},
    'SDWAN': {'Jitter_ms': 5, 'WAN_Path_Score': 90, 'BGP_Peer_Flaps': 0.5},
}

def identify_top_feature(link_type, metrics):
    baseline = HEALTHY_BASELINES.get(link_type, {})
    deviations = {}
    for feature, healthy_val in baseline.items():
        live_val = metrics.get(feature)
        if live_val is None:
            continue
        if feature == 'WAN_Path_Score':
            # lower is worse for this one, invert
            deviations[feature] = max(0, healthy_val - live_val) / healthy_val
        else:
            deviations[feature] = abs(live_val - healthy_val) / max(healthy_val, 1)

    if not deviations:
        return "Interface_Drop_Rate_Pct"

    return max(deviations, key=deviations.get)

# ─────────────────────────────────────────────
# 4. BUILD THE COPILOT PROMPT AND CALL OFFLINE LLM
# ─────────────────────────────────────────────
def generate_remediation(src_node, dst_node, link_type, metrics, prediction_confidence=None):
    top_feature = identify_top_feature(link_type, metrics)
    context = retrieve_context(link_type, top_feature)

    metrics_str = ", ".join(
        f"{k}: {v}" for k, v in metrics.items() if v is not None
    )

    confidence_str = f" (model confidence: {prediction_confidence:.0%})" if prediction_confidence else ""

    prompt = f"""You are a Network Operations Center (NOC) copilot for an air-gapped ISRO network.

A predictive model has flagged an imminent failure{confidence_str} on the link {src_node} -> {dst_node} ({link_type}).

Live telemetry: {metrics_str}
Primary anomaly detected in: {top_feature}

Relevant runbook context:
{context}

Based on the runbook above and the live telemetry, give a concise diagnosis (1 sentence) and exactly 3 short, actionable remediation steps. Be specific to this link type and the anomaly detected. Do not repeat the raw telemetry numbers back verbatim, interpret them."""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])

    return response['message']['content']

# ─────────────────────────────────────────────
# 5. STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Example: simulate a flagged SD-WAN link, same shape of data app.py will pass in
    test_metrics = {
        'RSVP_TE_Latency_ms': None,
        'OSPF_Cost_Metric': None,
        'Jitter_ms': 88.0,
        'WAN_Path_Score': 61.0,
        'BGP_Peer_Flaps': 3,
        'IPSec_Tunnel_Status': 1,
        'Interface_Drop_Rate_Pct': 2.8,
    }

    print("Asking the offline LLM Copilot...\n")
    result = generate_remediation(
        src_node="Hub-Core",
        dst_node="Overseas-East",
        link_type="SDWAN",
        metrics=test_metrics,
        prediction_confidence=0.91
    )
    print("AI Copilot Response:\n")
    print(result)
