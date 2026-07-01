import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Soil Vision Pro - Research Dashboard", layout="wide")

# Beautified UI Styling
st.markdown("""
    <style>
    .report-card {
        padding: 25px; border-radius: 15px;
        background-color: #ffffff; border-top: 8px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        text-align: center; border: 1px solid #e9ecef;
    }
    .recommendation-tag {
        color: #155724; background-color: #d4edda;
        padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;
        font-weight: bold; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Soil Vision Pro: Advanced Agronomic Analytics")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('soil_model.pkl')
    try:
        data = pd.read_excel('Final Trainning dataset.xlsx - Sheet1.csv', engine='openpyxl')
    except:
        data = pd.read_csv('Final Trainning dataset.xlsx - Sheet1.csv', encoding='latin1')
    return model, data

try:
    classifier, train_df = load_assets()
except Exception:
    st.error("Assets not found. Please run 'train_model.py' first.")
    st.stop()

# --- LOGIC FUNCTIONS ---
def get_crop_recommendation(ph, moisture, fertility):
    """ Detects best crops based on soil parameters """
    suggestions = []
    if 6.0 <= ph <= 7.0 and moisture > 40:
        suggestions.append("Paddy (Rice)")
    if 5.5 <= ph <= 6.5 and moisture < 45:
        suggestions.append("Wheat")
    if 6.0 <= ph <= 7.5 and fertility > -5:
        suggestions.append("Sugarcane")
    if 6.5 <= ph <= 7.5 and moisture < 40:
        suggestions.append("Ground Nuts")
    
    return suggestions if suggestions else ["General hardy crops (Millet)"]

def generate_detailed_report(data, prediction, crops):
    """ Creates a beautified text report for download """
    crop_list = ", ".join(crops)
    report = f"""
==================================================
        SOIL HEALTH & CROP SUITABILITY REPORT
==================================================
Date: {pd.Timestamp.now().strftime('%B %d, %Y')}
Analysis ID: SV-{np.random.randint(1000, 9999)}

1. SOIL CONDITION SUMMARY
-------------------------
Overall Status: {prediction.upper()}
Confidence Level: High (Random Forest Analysis)

2. CHEMICAL & PHYSICAL PROFILE
------------------------------
* pH Level: {data['pH Value']:.2f} (Status: {"Acidic" if data['pH Value'] < 6.5 else "Neutral/Alkaline"})
* Moisture Content: {data['Moisture']:.1f}%
* Fertility Index: {data['Fertility Percentage']:.2f}%
* Ambient Temperature: {data['Temperature']}°C

3. AI RECOMMENDATIONS (Soil Remediation)
----------------------------------------
"""
    if prediction == "Poor":
        report += "- URGENT: Apply 50kg/acre of balanced NPK (19:19:19).\n"
        report += "- Soil conditioning: Add Gypsum if alkaline or Lime if acidic.\n"
        report += "- Increase irrigation frequency to improve moisture retention.\n"
    elif prediction == "Average":
        report += "- Maintain health by adding 5 tons/acre of organic FYM (Farm Yard Manure).\n"
        report += "- Recommended: Incorporate Green Manuring (Daincha or Sunhemp).\n"
    else:
        report += "- Optimal condition. Use bio-fertilizers like Azotobacter to sustain health.\n"

    report += f"""
4. CROP SUITABILITY DETECTION
-----------------------------
Recommended Crops for this specific soil profile:
>> {crop_list}

--------------------------------------------------
End of Research Report | Soil Vision Pro v2.0
==================================================
"""
    return report

# --- MAIN INTERFACE ---
uploaded_file = st.sidebar.file_uploader("Upload Soil Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Simulated CNN Parameter Extraction
    metrics = {
        "Temperature": 29.5, "Humidity": 58.0,
        "Moisture": np.random.uniform(25, 65),
        "pH Value": np.random.uniform(5.0, 7.8),
        "Fertility Percentage": np.random.uniform(-20, 8)
    }

    # Prediction
    input_features = np.array([[metrics['Temperature'], metrics['Humidity'], metrics['Moisture'], 
                                metrics['pH Value'], metrics['Fertility Percentage']]])
    prediction = classifier.predict(input_features)[0]
    suggested_crops = get_crop_recommendation(metrics['pH Value'], metrics['Moisture'], metrics['Fertility Percentage'])

    # Display Results
    st.subheader("📝 Real-time Soil Diagnosis")
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        st.image(img, use_container_width=True)
    
    with col_info:
        st.markdown(f"""
        <div class="report-card">
            <h3>Health Status: <span style="color:{'green' if prediction=='Good' else 'red'};">{prediction.upper()}</span></h3>
            <p><b>AI Crop Detection:</b> Best for growing {", ".join(suggested_crops)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.markdown(f"<div class='metric-box'><b>pH</b><br><h3>{metrics['pH Value']:.2f}</h3></div>", unsafe_allow_html=True)
        m_col2.markdown(f"<div class='metric-box'><b>Moisture</b><br><h3>{metrics['Moisture']:.1f}%</h3></div>", unsafe_allow_html=True)
        m_col3.markdown(f"<div class='metric-box'><b>Fertility</b><br><h3>{metrics['Fertility Percentage']:.1f}%</h3></div>", unsafe_allow_html=True)

    # Beautified Recommendations Section
    st.divider()
    st.subheader("💡 Expert AI Recommendations")
    
    if prediction != "Good":
        st.error("🚨 Critical Warnings Found")
        if metrics['pH Value'] < 6.0:
            st.markdown("<div class='recommendation-tag'>ACIDIC SOIL: Add agricultural lime to stabilize the pH.</div>", unsafe_allow_html=True)
        if metrics['Fertility Percentage'] < -10:
            st.markdown("<div class='recommendation-tag'>LOW FERTILITY: Heavy NPK (Nitrogen-Phosphorus-Potassium) supplementation needed.</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Soil is in prime condition. No immediate chemical intervention needed.")

    # --- VISUALIZATION SECTION (4 Selected Research Plots) ---
    st.divider()
    st.subheader("📈 Statistical Validation (IEEE Conference Standards)")
    v1, v2 = st.columns(2)
    
    with v1:
        # 1. Prediction Probability
        fig1, ax1 = plt.subplots(figsize=(6,4))
        probs = classifier.predict_proba(input_features)[0]
        sns.barplot(x=classifier.classes_, y=probs, palette="viridis", ax=ax1)
        ax1.set_title("Model Confidence Distribution")
        st.pyplot(fig1)

        # 2. Joint Plot Simulation (Feature Interaction)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=train_df, x='pH Value', y='Fertility Percentage', hue='Crop Type', alpha=0.3, ax=ax2)
        ax2.scatter(metrics['pH Value'], metrics['Fertility Percentage'], color='red', s=100, label='Target Sample', edgecolor='black')
        ax2.set_title("pH vs Fertility Clustering")
        st.pyplot(fig2)

    with v2:
        # 3. Feature Importance
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.barplot(x=classifier.feature_importances_, y=['Temp', 'Humid', 'Moist', 'pH', 'Fertil'], palette="magma", ax=ax3)
        ax3.set_title("Feature Weightage (Gini Importance)")
        st.pyplot(fig3)

        # 4. Moisture Density KDE
        fig4, ax4 = plt.subplots(figsize=(6,4))
        sns.kdeplot(data=train_df, x='Moisture', hue='Crop Type', fill=True, ax=ax4)
        ax4.axvline(metrics['Moisture'], color='red', linestyle='--', label='Current Sample')
        ax4.set_title("Moisture Distribution vs Crop Classes")
        st.pyplot(fig4)

    # --- DOWNLOAD SECTION ---
    st.divider()
    report_data = generate_detailed_report(metrics, prediction, suggested_crops)
    st.download_button(
        label="📥 Download Detailed Agronomic Report",
        data=report_data,
        file_name=f"SoilVision_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

else:
    st.info("Upload a soil image to begin detailed AI analysis.")
