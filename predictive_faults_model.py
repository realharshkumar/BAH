import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('info/network_data.csv')

# ─────────────────────────────────────────────
# 2. FEATURE PREP
# ─────────────────────────────────────────────
# MPLS rows have None for Jitter_ms / WAN_Path_Score
# SD-WAN rows have None for RSVP_TE_Latency_ms / OSPF_Cost_Metric
# Fill with 0 -> the model learns "0 in this column = not applicable to this link type"
# and Link_Type (one-hot) tells it which feature set is relevant.
numeric_cols = [
    'RSVP_TE_Latency_ms', 'OSPF_Cost_Metric',
    'Jitter_ms', 'WAN_Path_Score',
    'BGP_Peer_Flaps', 'IPSec_Tunnel_Status'
]
# NOTE: Interface_Drop_Rate_Pct is deliberately EXCLUDED from features.
# It's the column the label itself was derived from (failure_label/precursor
# severity both set drop-rate directly), so including it causes leakage —
# the model would just learn "drop rate > threshold" instead of learning
# the actual leading indicators (jitter, latency, BGP flaps) that precede
# a failure. Excluding it forces the model to be genuinely PREDICTIVE.
df[numeric_cols] = df[numeric_cols].fillna(0)

# One-hot encode Link_Type (MPLS / SDWAN) so model learns type-specific patterns
df_encoded = pd.get_dummies(df, columns=['Link_Type'], prefix='Type')

feature_cols = numeric_cols + [c for c in df_encoded.columns if c.startswith('Type_')]

X = df_encoded[feature_cols]
y = df_encoded['Link_Failure_Imminent']  # predictive label, includes precursor rows

print(f"Total rows: {len(df)}")
print(f"Features used: {feature_cols}")
print(f"Failure-imminent rate: {y.mean():.2%}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',   # don't ignore the minority failure class
    random_state=42
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n─── Evaluation ───")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Failure Imminent']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                 Predicted Healthy   Predicted Failure")
print(f"Actual Healthy        {cm[0][0]:<10}        {cm[0][1]:<10}")
print(f"Actual Failure        {cm[1][0]:<10}        {cm[1][1]:<10}")

tn, fp, fn, tp = cm.ravel()
recall_failure = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nRecall on Failure class: {recall_failure:.2%}  (this is the number that matters most — missed failures)")

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n─── Feature Importance ───")
print(importances.to_string())

# ─────────────────────────────────────────────
# 7. SAVE MODEL + FEATURE SCHEMA
# ─────────────────────────────────────────────
joblib.dump(model, 'network_model.pkl')
joblib.dump(feature_cols, 'model_features.pkl')  # so app.py knows exact column order

print("\n✅ Saved: network_model.pkl")
print("✅ Saved: model_features.pkl (feature column order for inference)")