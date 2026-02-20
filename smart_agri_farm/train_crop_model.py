import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

print("🌱 TRAINING CROP RECOMMENDATION MODEL")
print("="*60)

# Load dataset
df = pd.read_csv('dataset/Crop_recommendation.csv')
print(f"Dataset loaded: {df.shape[0]} rows")

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Preprocessing
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(f"✅ MODEL ACCURACY: {accuracy:.2f}%")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/crop_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("💾 Models saved successfully!")
print("🚀 Run: python app.py")
