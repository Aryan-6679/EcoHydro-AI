```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import requests
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION & PREMIUM UI ---
st.set_page_config(
    page_title="EcoHydro AI - Precision Agriculture Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Glassmorphism Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        padding: 20px 24px;
        border-radius: 16px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.5);
    }

    /* Metric Label & Value Styling */
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #f8fafc !important;
        font-size: 2.2rem;
    }
    
    /* Gradients */
    h1, h2, h3 { color: #f8fafc !important; }
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #10b981, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# --- MACHINE LEARNING MODEL SIMULATION ---
@st.cache_resource
def load_and_train_models():
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

def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()['current']
            return data['temperature_2m'], data['relative_humidity_2m'], data.get('precipitation', 0.0)
    except Exception:
        pass
    return 32.5, 65.0, 0.0

@st.cache_data
def load_farm_data():
    try:
        return pd.read_csv("vellore_farm_data.csv")
    except FileNotFoundError:
        st.error("Error: vellore_farm_data.csv not found. Please ensure it is in the same folder.")
        return pd.DataFrame()

vellore_farms = load_farm_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='margin-top: -10px; margin-bottom: 0px;'>EcoHydro AI</h2>", unsafe_allow_html=True)
    st.caption("Precision Agriculture Engine v2.0")
    
    st.divider()
    page = st.radio("Select Workspace Module", [
        "GIS Farm Telemetry",
        "AI Irrigation Scheduler",
        "Methane & Carbon Ledger",
        "TAM/SAM & Business Model",
    ])
    st.divider()

# --- PAGE 1: GIS TELEMETRY ---
if page == "GIS Farm Telemetry":
    st.markdown("<h1 class='gradient-text'>Vellore Region Precision GIS Telemetry</h1>", unsafe_allow_html=True)
    st.write("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations (FPOs).")
    
    st.write("")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Monitored Land", "4,280 Acres", "+12% this month")
    col_m2.metric("Freshwater Saved", "18.4M Liters", "38.5% vs Flood")
    col_m3.metric("Verified CO2 Offsets", "1,420 tCO2e", "AWD Method")
    col_m4.metric("Farmer Payout Pool", "INR 28.4 Lakhs", "85% to Farmers")
    
    st.markdown("### Interactive Farm Map View")
    if not vellore_farms.empty:
        def get_color(status):
            if status == "Optimal":
                return [16, 185, 129, 220]
            elif status == "Deficit":
                return [245, 158, 11, 220]
            else:
                return [239, 68, 68, 220]

        vellore_farms['color'] = vellore_farms['status'].apply(get_color)
        vellore_farms['radius'] = vellore_farms['acres'] * 90
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=2,
            get_radius="radius",
            pickable=True,
            auto_highlight=True
        )
        
        view_state = pdk.ViewState(
            latitude=12.92,
            longitude=79.10,
            zoom=10,
            pitch=25
        )
        
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{name}</b><br/>FPO: {fpo}<br/>Crop: {crop}<br/>Area: {acres} Acres<br/>Moisture: {moisture}% ({status})",
                "style": {
                    "backgroundColor": "#1e293b",
                    "color": "#f8fafc",
                    "borderRadius": "8px",
                    "padding": "12px",
                    "fontFamily": "Plus Jakarta Sans, sans-serif"
                }
            }
        )
        
        st.pydeck_chart(deck, width="stretch")
        
        st.write("")
        st.markdown("### Selected Field Telemetry Breakout")
        selected_farm_name = st.selectbox("Select Farm for Inspection", vellore_farms["name"], label_visibility="collapsed")
        
        farm = vellore_farms[vellore_farms["name"] == selected_farm_name].iloc[0]
        temp, hum, precip = get_live_weather(farm['lat'], farm['lon'])
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"**FPO Partner:**\n{farm['fpo']}")
                st.info(f"**Target Crop:**\n{farm['crop']}")
            with c2:
                st.warning(f"**Soil Moisture:**\n{farm['moisture']}% ({'Optimal' if farm['moisture'] >= 30 else 'Deficit'})")
                st.warning(f"**Acreage:**\n{farm['acres']} Acres")
            with c3:
                st.success(f"**Live Local Weather:**\n{temp} C, {hum}% Humidity")
                st.success(f"**AWD Compliance:**\nVerified Compliant")

# --- PAGE 2: AI IRRIGATION SCHEDULER ---
elif page == "AI Irrigation Scheduler":
    st.markdown("<h1 class='gradient-text'>XGBoost/RandomForest AI Irrigation Simulator</h1>", unsafe_allow_html=True)
    st.write("Dynamically calculates precise water requirements to eliminate flood irrigation waste.")
    
    if 'api_temp' not in st.session_state: st.session_state.api_temp = 34
    if 'api_hum' not in st.session_state: st.session_state.api_hum = 60
    if 'api_rain' not in st.session_state: st.session_state.api_rain = 5
    
    def fetch_api_data():
        t, h, r = get_live_weather(12.9716, 79.1589)
        st.session_state.api_temp = int(t)
        st.session_state.api_hum = int(h)
        st.session_state.api_rain = int(r)
        st.toast("Live Vellore Weather Synced")
    
    st.button("Auto-Fill Live Vellore Weather", on_click=fetch_api_data, width="stretch")
    st.write("")
    
    col_inputs, col_results = st.columns([1.2, 1])
    with col_inputs:
        with st.expander("Farm & Soil Parameters", expanded=True):
            crop_type = st.selectbox("Target Crop", ["Paddy Rice", "Sugarcane", "Groundnut", "Maize"])
            c_acres, c_soil, c_nitro = st.columns(3)
            acres = c_acres.number_input("Field Area (Acres)", value=10.0)
            soil_m = c_soil.slider("Soil Moisture (%)", 10, 80, 25)
            nitrogen = c_nitro.slider("Nitrogen (mg/kg)", 10, 140, 65)
            
        with st.expander("Live Weather Data", expanded=True):
            c_temp, c_hum, c_rain = st.columns(3)
            temp_in = c_temp.slider("Temp (C)", 20, 45, key='api_temp')
            humidity_in = c_hum.slider("Humidity (%)", 20, 95, key='api_hum')
            rain_in = c_rain.slider("Rain (mm)", 0, 50, key='api_rain')
        
    with col_results:
        input_data = pd.DataFrame([{'N': nitrogen, 'P': 40, 'K': 30, 'Temp': temp_in, 'Humidity': humidity_in, 'pH': 6.8, 'Rainfall': rain_in, 'Soil_Moisture': soil_m}])
        
        if rain_in > 15:
            st.error("Heavy Rain Forecast: Postpone irrigation! Natural rainfall will replenish soil moisture.")
            pred_water = 0
        else:
            pred_water = model.predict(input_data)[0] * acres
            
        flood_water = 8500 * acres
        water_saved = max(0, flood_water - pred_water)
        
        st.markdown("### Optimization Results")
        st.metric("Traditional Flood Volume", f"{int(flood_water):,} L/Day")
        
        if pred_water == 0:
            st.metric("EcoHydro AI Drip Volume", "0 L/Day (Rain Override)", "-100.0% Saved", delta_color="normal")
            progress_pct = 0
        else:
            st.metric("EcoHydro AI Drip Volume", f"{int(pred_water):,} L/Day", f"-{round((water_saved/flood_water)*100, 1)}% Saved", delta_color="normal")
            progress_pct = int((pred_water/flood_water)*100)
            
        st.progress(progress_pct, text=f"AI uses only {progress_pct}% of traditional flood water.")
        st.success(f"**Monthly Power/Pumping Cost Saved:** INR {int(water_saved * 30 * 0.012):,}")
    
    st.divider()
    c_chart, c_explain = st.columns(2)
    with c_chart:
        st.markdown("#### 7-Day Water Projection")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig_chart = px.bar(
            pd.DataFrame({'Day': days, 'Flood': [int(flood_water)]*7, 'AI Drip': [int(pred_water * f) for f in [1.0, 0.92, 1.05, 0.88, 0.95, 1.02, 0.90]]}), 
            x='Day', y=['Flood', 'AI Drip'], barmode='group',
            color_discrete_map={"Flood": "#ef4444", "AI Drip": "#10b981"}
        )
        fig_chart.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=300, legend=dict(yanchor="bottom", y=-0.5, xanchor="center", x=0.5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_chart, width="stretch")
            
    with c_explain:
        st.markdown("#### Explainable AI (XAI) Weights")
        fig_importance = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale="Tealgrn")
        fig_importance.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_importance, width="stretch")

# --- PAGE 3: CARBON LEDGER ---
elif page == "Methane & Carbon Ledger":
    st.markdown("<h1 class='gradient-text'>Alternate Wetting and Drying (AWD) Carbon Engine</h1>", unsafe_allow_html=True)
    st.write("Verifies methane emission reductions via satellite radar (SAR) to mint high-quality ESG carbon credits.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Credits Minted", "1,314 tCO2e", "+200 this week")
    c2.metric("Market Price per tCO2e", "INR 1,530", "+ INR 45 vs last month")
    c3.metric("Total Farmer Liquidity", "INR 20,10,420", "Paid Out")
    
    st.write("")
    st.markdown("### Verified FPO Carbon Credit Batches")
    ledger_data = pd.DataFrame([
        {"Batch ID": "#AWD-26-089", "FPO Name": "North Arcot Farmers", "Acres": 450, "Credits (tCO2e)": 810, "Value (INR)": 1245000, "Compliance": 98, "Status": "Paid"},
        {"Batch ID": "#AWD-26-090", "FPO Name": "Palar River Basin", "Acres": 280, "Credits (tCO2e)": 504, "Value (INR)": 774000, "Compliance": 100, "Status": "Paid"},
        {"Batch ID": "#AWD-26-091", "FPO Name": "Walajah Organics", "Acres": 150, "Credits (tCO2e)": 270, "Value (INR)": 413100, "Compliance": 92, "Status": "Pending Audit"}
    ])
    
    st.dataframe(
        ledger_data,
        column_config={
            "Batch ID": st.column_config.TextColumn("Batch ID", width="small"),
            "Value (INR)": st.column_config.NumberColumn("Payout Value", format="INR %d"),
            "Compliance": st.column_config.ProgressColumn("AWD Compliance %", format="%d%%", min_value=0, max_value=100),
            "Status": st.column_config.TextColumn("Status")
        },
        hide_index=True,
        width="stretch"
    )
    
    st.download_button("Download SEBI BRSR ESG Report (CSV)", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Carbon_Audit.csv', mime='text/csv', width="stretch")

# --- PAGE 4: BUSINESS MODEL ---
elif page == "TAM/SAM & Business Model":
    st.markdown("<h1 class='gradient-text'>Financial Projections & Market Sizing</h1>", unsafe_allow_html=True)
    st.write("EcoHydro AI leverages a dual-revenue stream: B2B SaaS for FPOs and a 15% marketplace commission on carbon credits.")
    
    st.write("")
    col_tam1, col_tam2, col_tam3 = st.columns(3)
    col_tam1.metric("TAM (Smart Ag & Offsets)", "$12.5 Billion", "Global")
    col_tam2.metric("SAM (Indian Irrigation Automation)", "$1.8 Billion", "South India Focus")
    col_tam3.metric("SOM (Year 3 Projection)", "$42 Million", "Serviceable Obtainable")
    
    st.write("---")
    st.markdown("### Interactive Monetization Calculator")
    
    calc_c1, calc_c2 = st.columns([1, 1.5])
    with calc_c1:
        st.info("Adjust the sliders to simulate Year-1 Revenue")
        fpos = st.slider("Onboarded FPOs", 5, 300, 30)
        credits = st.slider("Carbon Credits Traded (tCO2e)", 1000, 100000, 15000, step=1000)
        
        saas_rev = fpos * 2500 * 12
        carbon_rev = credits * 1650 * 0.15
        total_arr = saas_rev + carbon_rev
        
    with calc_c2:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 25px; border-radius: 16px;">
            <h4 style="margin-top: 0; color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Projected Annual Recurring Revenue (ARR)</h4>
            <h1 style="color: #10b981; font-size: 3rem; margin: 10px 0;">INR {(total_arr)/100000:.2f} Lakhs/Yr</h1>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 1.1rem;">
                <span><b>SaaS Subscriptions:</b> INR {saas_rev/100000:.2f}L</span>
                <span><b>Carbon Commission:</b> INR {carbon_rev/100000:.2f}L</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
```:app.py
```
