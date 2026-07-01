"""
predictive_faults.py
Predicts network link failure before it happens using an LSTM on sliding windows
of link telemetry.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers

WINDOW_SIZE = 5
DATA_PATH = "info/network_data.csv"

# ---------------------------------------------------------------------------
# 1. Load + preprocess
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["Source_Node", "Dest_Node", "Timestamp"]).reset_index(drop=True)

# fill nulls (MPLS-only / SDWAN-only cols)
null_fill_cols = [
    "RSVP_TE_Latency_ms", "OSPF_Cost_Metric", "Jitter_ms", "WAN_Path_Score"
]
df[null_fill_cols] = df[null_fill_cols].fillna(0)

# one-hot encode Link_Type
df = pd.get_dummies(df, columns=["Link_Type"], prefix="Link_Type")

# feature columns: exclude ids/time/label/symptom col
exclude_cols = {
    "Timestamp", "Source_Node", "Dest_Node", "Link_Failed",
    "Interface_Drop_Rate_Pct",
}
feature_cols = [c for c in df.columns if c not in exclude_cols]

# ensure numeric/bool -> float
for c in feature_cols:
    df[c] = df[c].astype(float)

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# ---------------------------------------------------------------------------
# 2. Sliding windows per link, label = Link_Failed of row AFTER the window
# ---------------------------------------------------------------------------
X_windows, y_labels = [], []

for (_src, _dst), group in df.groupby(["Source_Node", "Dest_Node"]):
    group = group.reset_index(drop=True)
    feats = group[feature_cols].values
    labels = group["Link_Failed"].values

    # window covers rows [i, i+WINDOW_SIZE-1], label is row i+WINDOW_SIZE
    for i in range(len(group) - WINDOW_SIZE):
        X_windows.append(feats[i:i + WINDOW_SIZE])
        y_labels.append(labels[i + WINDOW_SIZE])

X = np.array(X_windows, dtype=np.float32)
y = np.array(y_labels, dtype=np.float32)

print(f"X shape: {X.shape}  y shape: {y.shape}  positive rate: {y.mean():.4f}")

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# class weights for imbalance
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): w for c, w in zip(classes, weights)}
print(f"Class weights: {class_weight}")

# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
n_features = X.shape[2]

model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, n_features)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=2,
)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
y_prob = model.predict(X_test).ravel()

# ---------------------------------------------------------------------------
# 5a. Threshold sweep (precision/recall tradeoff)
# ---------------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

print("\nThreshold sweep:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10}")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (y_prob >= t).astype(int)
    tp = ((pred == 1) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"{t:>10.1f} {prec:>10.2%} {rec:>10.2%}")

# pick best threshold: highest recall among thresholds with precision >= 0.2
# pick highest threshold that keeps precision >= 0.2 and recall >= 0.8
candidates = [(p, r, t) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if p >= 0.9 and r >= 0.9 and t < 1.0]
if candidates:
    best_p, best_r, best_t = min(candidates, key=lambda x: x[2])
else:
    best_t = 0.5
    best_p, best_r = None, None
print(f"\nChosen threshold: {best_t:.2f}" + (f" (precision={best_p:.2%}, recall={best_r:.2%})" if best_p else " (fallback, no threshold hit precision>=20%)"))

y_pred = (y_prob >= best_t).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# 6. Save model + feature list
# ---------------------------------------------------------------------------
model.save("network_model.keras")
joblib.dump(feature_cols, "model_features.pkl")
joblib.dump(best_t, "model_threshold.pkl")

print("\nSaved network_model.keras, model_features.pkl, model_threshold.pkl")
