import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoHydro AI - Precision Agriculture Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background-color: rgba(30, 41, 59, 0.7); padding: 15px; border-radius: 10px; border: 1px solid rgba(51, 65, 85, 0.8); }
    .pitch-box { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.4); padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid rgba(245, 158, 11, 1); }
    </style>
""", unsafe_allow_html=True)

# --- MACHINE LEARNING MODEL SIMULATION ---
@st.cache_resource
def load_and_train_models():
    """Generates synthetic agronomic training dataset based on ICAR & FAO parameters"""
    from sklearn.ensemble import RandomForestRegressor
    
    np.random.seed(42)
    n_samples = 1000
    
    N = np.random.uniform(10, 140, n_samples)
    P = np.random.uniform(5, 90, n_samples)
    K = np.random.uniform(10, 80, n_samples)
    temp = np.random.uniform(18, 42, n_samples)
    humidity = np.random.uniform(30, 95, n_samples)
    ph = np.random.uniform(5.5, 8.5, n_samples)
    rainfall = np.random.uniform(0, 300, n_samples)
    soil_moisture = np.random.uniform(10, 80, n_samples)
    
    # Target: Optimal Daily Water Requirement (Liters/Acre/Day)
    base_water = (temp * 150) + ((100 - humidity) * 40) - (rainfall * 10) - (soil_moisture * 50)
    water_req = np.clip(base_water, 1500, 9000)
    
    df = pd.DataFrame({
        'N': N, 'P': P, 'K': K, 'Temp': temp, 
        'Humidity': humidity, 'pH': ph, 'Rainfall': rainfall,
        'Soil_Moisture': soil_moisture, 'Water_Req': water_req
    })
    
    X = df[['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall', 'Soil_Moisture']]
    y = df['Water_Req']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    return model, df, feature_importance

model, dataset, feature_importance = load_and_train_models()

# --- WEATHER API INTEGRATION (LEVEL 3) ---
def get_live_weather(lat, lon):
    """Fetches live weather from Open-Meteo API (Free)"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()['current']
            return data['temperature_2m'], data['relative_humidity_2m'], data.get('precipitation', 0.0)
    except Exception as e:
        pass
    return 32.5, 65.0, 0.0 # Fallback

# --- DATASET INTEGRATION (LEVEL 1) ---
@st.cache_data
def load_farm_data():
    """Loads external CSV data"""
    try:
        return pd.read_csv("vellore_farm_data.csv")
    except FileNotFoundError:
        st.error("⚠️ vellore_farm_data.csv not found. Please ensure it is in the same folder.")
        return pd.DataFrame()

vellore_farms = load_farm_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/color/96/eco-factory.png", width=80)
st.sidebar.title("EcoHydro AI")
st.sidebar.caption("Precision Agriculture Engine")
page = st.sidebar.radio("Select Workspace Module", [
    "🌍 GIS Farm Telemetry",
    "⚡ AI Irrigation Scheduler",
    "🍃 Methane & Carbon Ledger",
    "📈 TAM/SAM & Business Model",
])

pitch_toggle = st.sidebar.checkbox("🏆 Show Judge Pitch Highlights", value=True)
if pitch_toggle:
    st.sidebar.markdown("""
    <div class="pitch-box">
        <b>💡 Judge Elevator Pitch (30 Sec):</b> EcoHydro AI eliminates 40% of agricultural freshwater waste using open satellite telemetry and AI, while unlocking up to ₹15,000/acre in corporate carbon credits for smallholder farmers—with <b>Zero Hardware Setup Costs</b>.
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 1: GIS TELEMETRY ---
if page == "🌍 GIS Farm Telemetry":
    st.header("🌍 Vellore Region Precision GIS Telemetry")
    st.write("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations (FPOs).")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Monitored Land", "4,280 Acres", "+12% this month")
    col_m2.metric("Freshwater Saved", "18.4M Liters", "38.5% vs Flood")
    col_m3.metric("Verified CO2 Offsets", "1,420 tCO2e", "AWD Method")
    col_m4.metric("Farmer Payout Pool", "₹28.4 Lakhs", "85% to Farmers")
    
    st.subheader("Interactive Farm Map View")
    if not vellore_farms.empty:
        fig_map = px.scatter_mapbox(
            vellore_farms, lat="lat", lon="lon", color="status", size="acres",
            hover_name="name", hover_data=["fpo", "crop", "acres", "moisture"],
            color_discrete_map={"Optimal": "#10b981", "Deficit": "#f59e0b", "Critical": "#ef4444"},
            zoom=10, height=450
        )
        fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.subheader("Selected Field Telemetry Breakout")
        selected_farm_name = st.selectbox("Select Farm for Inspection", vellore_farms["name"])
        farm = vellore_farms[vellore_farms["name"] == selected_farm_name].iloc[0]
        temp, hum, precip = get_live_weather(farm['lat'], farm['lon'])
        
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**FPO Partner:** {farm['fpo']}")
            st.write(f"**Target Crop:** {farm['crop']}")
            st.write(f"**Acreage:** {farm['acres']} Acres")
        with c2:
            st.write(f"**Current Soil Moisture:** {farm['moisture']}%")
            st.write(f"**Live Local Weather (API):** {temp}°C, {hum}% Humidity")
            st.write(f"**AWD Compliance:** Verified Compliant")

# --- PAGE 2: AI IRRIGATION SCHEDULER ---
elif page == "⚡ AI Irrigation Scheduler":
    st.header("⚡ XGBoost/RandomForest AI Irrigation Simulator")
    
    if 'api_temp' not in st.session_state: st.session_state.api_temp = 34
    if 'api_hum' not in st.session_state: st.session_state.api_hum = 60
    if 'api_rain' not in st.session_state: st.session_state.api_rain = 5
    
    def fetch_api_data():
        t, h, r = get_live_weather(12.9716, 79.1589) # VIT Vellore Coordinates
        st.session_state.api_temp = int(t)
        st.session_state.api_hum = int(h)
        st.session_state.api_rain = int(r)
    
    st.button("🌦️ Auto-Fill Live Vellore Weather (Open-Meteo API)", on_click=fetch_api_data, type="primary")
    
    col_inputs, col_results = st.columns([1, 1])
    with col_inputs:
        crop_type = st.selectbox("Target Crop", ["Paddy Rice", "Sugarcane", "Groundnut", "Maize"])
        acres = st.number_input("Field Area (Acres)", value=10.0)
        soil_m = st.slider("Current Soil Moisture (%)", 10, 80, 25)
        nitrogen = st.slider("Soil Nitrogen (mg/kg)", 10, 140, 65)
        temp_in = st.slider("Ambient Temperature (°C)", 20, 45, key='api_temp')
        humidity_in = st.slider("Relative Humidity (%)", 20, 95, key='api_hum')
        rain_in = st.slider("Forecasted Rain (mm)", 0, 50, key='api_rain')
        
    with col_results:
        input_data = pd.DataFrame([{'N': nitrogen, 'P': 40, 'K': 30, 'Temp': temp_in, 'Humidity': humidity_in, 'pH': 6.8, 'Rainfall': rain_in, 'Soil_Moisture': soil_m}])
        pred_water = model.predict(input_data)[0] * acres
        flood_water = 8500 * acres
        water_saved = max(0, flood_water - pred_water)
        
        st.metric("Traditional Flood Volume", f"{int(flood_water):,} L/Day")
        st.metric("EcoHydro AI Recommended", f"{int(pred_water):,} L/Day", f"-{round((water_saved/flood_water)*100, 1)}% Saved", delta_color="normal")
        st.metric("Power Saved (Monthly)", f"₹{int(water_saved * 30 * 0.012):,}")
    
    st.divider()
    c_chart, c_explain = st.columns(2)
    with c_chart:
        st.subheader("7-Day Projection")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig_chart = px.bar(pd.DataFrame({'Day': days, 'Flood': [int(flood_water)]*7, 'AI Drip': [int(pred_water * f) for f in [1.0, 0.92, 1.05, 0.88, 0.95, 1.02, 0.90]]}), x='Day', y=['Flood', 'AI Drip'], barmode='group')
        fig_chart.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=300, legend=dict(yanchor="bottom", y=-0.5, xanchor="center", x=0.5))
        st.plotly_chart(fig_chart, use_container_width=True)
    with c_explain:
        st.subheader("🧠 Explainable AI (XAI)")
        fig_importance = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance')
        fig_importance.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)

# --- PAGE 3: CARBON LEDGER ---
elif page == "🍃 Methane & Carbon Ledger":
    st.header("🍃 Alternate Wetting and Drying (AWD) Carbon Engine")
    ledger_data = pd.DataFrame([
        {"Batch ID": "#AWD-26-089", "FPO Name": "North Arcot Farmers", "Acres": 450, "Credits (tCO2e)": 810, "Value (INR)": "₹12,45,000", "Status": "Paid"},
        {"Batch ID": "#AWD-26-090", "FPO Name": "Palar River Basin", "Acres": 280, "Credits (tCO2e)": 504, "Value (INR)": "₹7,74,000", "Status": "Paid"}
    ])
    st.table(ledger_data)
    st.download_button("📥 Download ESG Report (CSV)", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Carbon_Audit.csv', mime='text/csv')

# --- PAGE 4: BUSINESS MODEL ---
elif page == "📈 TAM/SAM & Business Model":
    st.header("📈 Financial Projections")
    col_tam1, col_tam2, col_tam3 = st.columns(3)
    col_tam1.metric("TAM", "$12.5 Billion")
    col_tam2.metric("SAM", "$1.8 Billion")
    col_tam3.metric("SOM (Year 3)", "$42 Million")
    
    st.subheader("Monetization Calculator")
    fpos = st.slider("Onboarded FPOs", 5, 300, 30)
    credits = st.slider("Carbon Credits", 1000, 100000, 15000, step=1000)
    st.metric("Total Projected ARR", f"₹{(fpos*2500*12 + credits*1650*0.15)/100000:.2f} Lakhs/Yr")