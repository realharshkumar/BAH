import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the fake data you generated
df = pd.read_csv('network_data.csv')

# 2. Select your inputs (Features) and output (Target)
X = df[['Latency_ms', 'Packet_Loss_Pct', 'Bandwidth_Usage_Pct']]
y = df['Link_Failure']

# 3. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest model
print("Training the predictor model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the trained model to your laptop as a file
joblib.dump(model, 'network_model.pkl')

print("Success: 'network_model.pkl' has been saved completely offline!")