import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os

# --- 1. UPDATED DATA LOADING ---
file_path = 'Final Trainning dataset.xlsx - Sheet1.csv'

try:
    # Use read_excel because the file content is binary (XLSX format)
    # engine='openpyxl' is required for .xlsx files
    df = pd.read_excel(file_path, engine='openpyxl')
    print("✅ Success: Loaded binary Excel data.")
except Exception as e:
    print(f"Excel load failed: {e}. Trying CSV fallback...")
    # Last resort fallback if it really is a CSV with complex characters
    df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')

# --- 2. DATA PREPARATION ---
def categorize_soil(row):
    ph, fertility = row['pH Value'], row['Fertility Percentage']
    if (6.0 <= ph <= 7.5) and (fertility > -10): return "Good"
    elif (5.5 <= ph <= 8.0) and (fertility > -20): return "Average"
    else: return "Poor"

df['Soil Condition'] = df.apply(categorize_soil, axis=1)

# Ensure column names match your Excel sheet exactly
features = ['Temparature', 'Humidity', 'Moisture', 'pH Value', 'Fertility Percentage']
X = df[features]
y = df['Soil Condition']

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 3. SAVE MODEL & PLOTS ---
joblib.dump(model, 'soil_model.pkl')
print("✅ Success: 'soil_model.pkl' created.")

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')

plt.subplot(2, 1, 2)
importances = model.feature_importances_
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance for Soil Health')

plt.tight_layout()
plt.savefig('ieee_research_plots.png', dpi=300)
print("✅ Success: Plots saved for IEEE paper.")
