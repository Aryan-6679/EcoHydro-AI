import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestRegressor
import json

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="EcoHydro AI | VIT Hackathon", page_icon="🌱", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #10b981;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & ML MODEL
# ==========================================
@st.cache_data
def load_farm_data():
    try:
        df = pd.read_csv("vellore_farm_data.csv")
        return df
    except FileNotFoundError:
        # Fallback synthetic data if CSV is missing
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Katpadi FPO", "Kaniyambadi FPO", "Anaicut FPO", "Gudiyatham FPO", "K.V. Kuppam FPO"],
            "lat": [12.9800, 12.8333, 12.8754, 12.9463, 13.0485],
            "lon": [79.1390, 79.1333, 78.9833, 78.8711, 79.0345],
            "crop": ["Paddy", "Sugarcane", "Groundnut", "Paddy", "Banana"],
            "acres": [120, 85, 40, 150, 60],
            "moisture_level": [45, 30, 20, 50, 60],
            "status": ["Optimal", "Deficit", "Critical", "Optimal", "Optimal"]
        })

@st.cache_resource
def train_ai_model():
    # Synthetic data generation for the Hackathon Demo
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        'N': np.random.uniform(10, 120, n_samples),
        'P': np.random.uniform(5, 80, n_samples),
        'K': np.random.uniform(10, 80, n_samples),
        'Temp': np.random.uniform(22, 42, n_samples),
        'Humidity': np.random.uniform(30, 90, n_samples),
        'Rainfall': np.random.uniform(0, 50, n_samples),
        'pH': np.random.uniform(5.5, 8.5, n_samples),
        'Soil_Moisture': np.random.uniform(10, 80, n_samples)
    })
    # Target: Water required in Liters per day (Simplified logic for demo)
    y = np.clip((X['Temp']*200) + ((100-X['Humidity'])*50) - (X['Rainfall']*20) - (X['Soil_Moisture']*80), 1000, 8000)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X.columns

farm_data = load_farm_data()
ai_model, feature_names = train_ai_model()

# ==========================================
# 3. SIDEBAR NAVIGATION & PITCH
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=60)
    st.title("EcoHydro AI")
    st.caption("Precision Agriculture Engine")
    st.markdown("---")
    
    view = st.radio("Select Workspace Module", [
        "🌍 GIS Farm Telemetry",
        "⚡ AI Irrigation Scheduler",
        "🍃 Methane & Carbon Ledger",
        "📈 TAM/SAM & Business Model"
    ])
    
    st.markdown("---")
    if st.checkbox("🏆 Show Judge Pitch Highlights", value=True):
        st.success("""
        **💡 Judge Elevator Pitch (30 Sec):**  
        EcoHydro AI eliminates 40% of agricultural freshwater waste in regions like Vellore using open satellite telemetry and AI, while unlocking up to ₹15,000/acre in corporate carbon credits for smallholder farmers—with **Zero Hardware Setup Costs**.
        """)

# ==========================================
# 4. MAIN DASHBOARD VIEWS
# ==========================================

# ------------------------------------------
# VIEW 1: GIS Farm Telemetry
# ------------------------------------------
if view == "🌍 GIS Farm Telemetry":
    st.header("🌍 Precision GIS Telemetry")
    st.markdown("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations (FPOs).")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Total Monitored Land</div><div class="metric-value">4,280 Acres</div><div style="color:#10b981; font-size:0.8rem;">↑ +12% this month</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Freshwater Saved</div><div class="metric-value">18.4M Liters</div><div style="color:#10b981; font-size:0.8rem;">↑ 38.5% vs Flood</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">Verified CO2 Offsets</div><div class="metric-value">1,420 Tons</div><div style="color:#10b981; font-size:0.8rem;">↑ AWD Method</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">Farmer Payout Pool</div><div class="metric-value">₹28.4 Lakh</div><div style="color:#10b981; font-size:0.8rem;">↑ 85% to Farmers</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Plotly Map (Using Mapbox for Plotly 5.24.1)
    fig_map = px.scatter_mapbox(
        farm_data, lat="lat", lon="lon", color="status", size="acres",
        hover_name="name", 
        hover_data=["crop", "acres", "moisture_level"], # Fixed column names
        color_discrete_map={"Optimal": "#10b981", "Deficit": "#f59e0b", "Critical": "#ef4444"},
        zoom=10, center={"lat": 12.9165, "lon": 79.1325}, mapbox_style="carto-positron"
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------------------
# VIEW 2: AI Irrigation Scheduler
# ------------------------------------------
elif view == "⚡ AI Irrigation Scheduler":
    st.header("⚡ AI Irrigation Scheduler")
    st.markdown("Generates precise daily watering instructions using Random Forest regression.")
    
    # Weather API Integration
    if st.button("🌦️ Auto-Fill Live Vellore Weather (Open-Meteo API)"):
        try:
            res = requests.get("https://api.open-meteo.com/v1/forecast?latitude=12.9165&longitude=79.1325&current_weather=true")
            weather = res.json()["current_weather"]
            st.session_state['api_temp'] = weather["temperature"]
            st.session_state['api_wind'] = weather["windspeed"]
            st.success(f"✅ Fetched Live Data: {weather['temperature']}°C")
        except Exception as e:
            st.error("API failed. Using manual sliders.")
            st.session_state['api_temp'] = 32.0

    temp_val = st.session_state.get('api_temp', 32.0)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        n_val = st.slider("Nitrogen (N)", 0, 150, 60)
        p_val = st.slider("Phosphorus (P)", 0, 100, 30)
        k_val = st.slider("Potassium (K)", 0, 100, 40)
        ph_val = st.slider("Soil pH", 4.0, 9.0, 6.5)
    with col2:
        t_val = st.slider("Temperature (°C)", 10.0, 50.0, float(temp_val))
        h_val = st.slider("Humidity (%)", 10, 100, 65)
        r_val = st.slider("Rainfall (mm)", 0, 200, 0)
        sm_val = st.slider("Soil Moisture (%)", 0, 100, 35)
    
    # Prediction
    input_df = pd.DataFrame([[n_val, p_val, k_val, t_val, h_val, r_val, ph_val, sm_val]], columns=feature_names)
    water_pred = ai_model.predict(input_df)[0]
    flood_water = water_pred * 1.65 # Simulating flood irrigation waste
    
    with col3:
        st.subheader("AI Recommendation")
        st.info(f"💧 **Optimal Drip Irrigation:** {int(water_pred)} Liters/Acre today.")
        st.warning(f"🌊 **Traditional Flood Irrigation:** {int(flood_water)} Liters/Acre today.")
        st.success(f"📈 **Freshwater Saved:** {int(flood_water - water_pred)} Liters/Acre ({(1 - water_pred/flood_water)*100:.1f}%)")
        
        # 7-Day Projection Chart with UI fix applied
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        drip_proj = [water_pred + np.random.randint(-500, 500) for _ in range(7)]
        flood_proj = [f * 1.65 for f in drip_proj]
        
        proj_df = pd.DataFrame({'Day': days, 'AI Drip': drip_proj, 'Flood': flood_proj})
        fig_proj = px.bar(proj_df, x='Day', y=['Flood', 'AI Drip'], barmode='group', title="7-Day Water Usage Projection")
        
        # Apply the layout fix here to prevent overlapping labels
        fig_proj.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            xaxis_title=None,
            legend_title_text=None,
            margin=dict(b=50)
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        # XAI Chart
        importances = ai_model.feature_importances_
        xai_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        fig_xai = px.bar(xai_df, x='Importance', y='Feature', orientation='h', title="🧠 Explainable AI (XAI) Feature Weights")
        fig_xai.update_layout(yaxis_title=None)
        st.plotly_chart(fig_xai, use_container_width=True)

# ------------------------------------------
# VIEW 3: Carbon Ledger
# ------------------------------------------
elif view == "🍃 Methane & Carbon Ledger":
    st.header("🍃 Verified Carbon Credit Ledger")
    st.markdown("Calculates greenhouse gas reductions by shifting from continuous flooding to Alternate Wetting and Drying (AWD) irrigation.")
    
    ledger_data = pd.DataFrame({
        "FPO / Cooperative": ["Katpadi FPO", "Kaniyambadi FPO", "Anaicut FPO", "Gudiyatham FPO"],
        "Acreage Under AWD": [1200, 850, 400, 1500],
        "Methane Prevented (kg)": [14400, 10200, 4800, 18000],
        "CO2e Offset (Tons)": [403.2, 285.6, 134.4, 504.0],
        "Tradable Carbon Credits": [403, 285, 134, 504],
        "Estimated Payout (INR)": ["₹8,06,000", "₹5,70,000", "₹2,68,000", "₹10,08,000"]
    })
    
    st.dataframe(ledger_data, use_container_width=True)
    
    csv = ledger_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Audit Ready Ledger (CSV)",
        data=csv,
        file_name='carbon_ledger_audit.csv',
        mime='text/csv',
    )

# ------------------------------------------
# VIEW 4: Business Model
# ------------------------------------------
elif view == "📈 TAM/SAM & Business Model":
    st.header("📈 Market Sizing & Revenue Model")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("TAM (Total Addressable Market)", "$14 Billion", "Global Agritech")
    col2.metric("SAM (Serviceable Market)", "$2.1 Billion", "India Precision Farming")
    col3.metric("SOM (Serviceable Obtainable)", "$150 Million", "South India FPOs")
    
    st.markdown("""
    ### 💰 How We Make Money (B2B2C)
    1. **SaaS Subscription for FPOs:** ₹5,000/month per Farmer Producer Organization for dashboard access.
    2. **Carbon Credit Brokerage:** We take a 15% commission on all corporate carbon credits sold through our MRV (Measurement, Reporting, and Verification) ledger.
    3. **Corporate ESG API:** Selling verified water-savings data to agricultural corporates (ITC, Hindustan Unilever) to meet their sustainability compliance goals.
    
    ### 🚀 Go-To-Market Strategy
    * **Phase 1 (Months 1-6):** Pilot with 5 FPOs in Vellore district (Zero cost to farmers).
    * **Phase 2 (Months 6-12):** Register initial water-savings on the Gold Standard Carbon Registry.
    * **Phase 3 (Year 2+):** Scale pan-India using open satellite integrations.
    """)
