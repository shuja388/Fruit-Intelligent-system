import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="FWIS - Fruit Waste Intelligence System",
    page_icon="🍏",
    layout="centered"
)

# --- 1. Load Models & Column Layout ---
@st.cache_resource
def load_assets():
    try:
        model_waste = joblib.load('model_waste.pkl')
        model_rsl = joblib.load('model_rsl.pkl')
        model_reason = joblib.load('model_reason.pkl')
        columns_layout = joblib.load('columns_layout.pkl')
        return model_waste, model_rsl, model_reason, columns_layout
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Ensure all .pkl files are in the same folder.")
        st.stop()

model_waste, model_rsl, model_reason, columns_layout = load_assets()

# --- 2. Title and Description ---
st.title("🍏 Fruit Waste Intelligence System (FWIS)")
st.markdown("""
Predict transit waste, remaining shelf life, and pinpoint primary quality risks for inbound manifests before they hit your distribution bay.
""")
st.write("---")

# --- 3. UI Inputs ---
st.subheader("📋 Inbound Shipment Profile")

col1, col2 = st.columns(2)

with col1:
    fruit = st.selectbox(
        "Commodity Type", 
        ['Mango', 'Apple', 'Banana', 'Strawberry', 'Orange']
    )
    packaging = st.selectbox(
        "Packaging Type", 
        ['Plastic Vent', 'Corrugated Carton', 'Heavy Wooden', 'Mesh Bag']
    )
    temp = st.slider("Average Transit Temperature (°C)", 0.0, 30.0, 15.0, 0.5)

with col2:
    humidity = st.slider("Relative Humidity (%)", 50.0, 100.0, 75.0, 0.5)
    distance = st.number_input("Transit Distance (km)", min_value=100.0, max_value=5000.0, value=1000.0, step=50.0)
    delay = st.number_input("Unplanned Delay (Days)", min_value=0, max_value=15, value=0, step=1)

# --- 4. Feature Engineering & Pipeline Alignment ---
# Replicate training math exactly
base_transit_days = distance / 450.0
transit_days = base_transit_days + delay
degree_days = temp * transit_days
vpd_proxy = (100.0 - humidity) * temp

# Construct input row dictionary initialized to zero
input_data = {col: 0.0 for col in columns_layout}

# Fill continuous variables
input_data['Transit_Temp_C'] = temp
input_data['RH_Pct'] = humidity
input_data['Distance_km'] = distance
input_data['Delay_Days'] = delay
input_data['Transit_Days'] = transit_days
input_data['Degree_Days'] = degree_days
input_data['VPD_Proxy'] = vpd_proxy

# Handle categorical variables (One-Hot Encoding with drop_first=True alignment)
fruit_col = f"Fruit_{fruit}"
if fruit_col in input_data:
    input_data[fruit_col] = 1.0

pkg_col = f"Packaging_Type_{packaging}"
if pkg_col in input_data:
    input_data[pkg_col] = 1.0

# Convert to DataFrame matching model columns exactly
features_df = pd.DataFrame([input_data])[columns_layout]

# --- 5. Predictions & UI Outputs ---
st.write("---")
if st.button("🔮 Run FWIS Diagnostics", type="primary"):
    
    # Generate predictions
    pred_w = model_waste.predict(features_df)[0]
    pred_r = model_rsl.predict(features_df)[0]
    pred_msg = model_reason.predict(features_df)[0]
    
    st.subheader("📊 FWIS Intelligence Alert")
    
    # Metric Callouts
    m1, m2 = st.columns(2)
    m1.metric(label="Expected Batch Waste", value=f"{pred_w:.1f}%")
    m2.metric(label="Remaining Shelf Life", value=f"{pred_r:.1f} Days")
    
    # Risk Box
    st.warning(f"⚠️ **Primary Degradation Risk:** {pred_msg}")
    
    # Action Logic Engine
    st.subheader("⚡ Recommended Operations Routing")
    if pred_w > 25.0 or pred_r < 3.0:
        st.error(
            "🚨 **ACTION REQUIRED:** Route instantly to local discount processing or juice processing. "
            "**Do NOT export.**"
        )
    elif pred_r <= 7.0:
        st.info(
            "⚡ **ACTION REQUIRED:** 'First-Expired, First-Out' (FEFO) override triggered. "
            "Route to nearest regional market."
        )
    else:
        st.success(
            "✅ **ACTION REQUIRED:** Safe for standard cold storage or long-distance redistribution chains."
        )
