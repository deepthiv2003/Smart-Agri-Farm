import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

print("🌧️ TRAINING RAINFALL MODEL")
df = pd.read_csv('dataset/rainfall in india 1901-2015.csv')

# Karnataka data only
karnataka_data = df[df['SUBDIVISION'].str.contains('KARNATAKA', na=False)]

# Process monthly data
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
data = []
for _, row in karnataka_data.iterrows():
    for i, month in enumerate(months):
        if pd.notna(row[month]):
            data.append([row['YEAR'], i+1, float(row[month])])

df_rain = pd.DataFrame(data, columns=['year', 'month', 'rainfall'])
X = df_rain[['year', 'month']]
y = df_rain['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"✅ Rainfall MAE: {mae:.2f}mm")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rainfall_model.pkl')
print("💾 Rainfall model saved!")
