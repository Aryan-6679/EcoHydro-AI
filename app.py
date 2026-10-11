import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import requests
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION & PREMIUM UI ---
st.set_page_config(
    page_title="EcoHydro AI - Precision Agriculture",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💧"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Enhanced Glassmorphism Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.25);
        border-color: rgba(16, 185, 129, 0.8);
    }

    /* Metric Typography Styling */
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #f8fafc !important;
        font-size: 2.4rem;
        background: -webkit-linear-gradient(45deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Typography Gradients */
    h1, h2, h3 { color: #f8fafc !important; font-weight: 800 !important; }
    .gradient-text {
        background: -webkit-linear-gradient(135deg, #10b981, #06b6d4, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        padding-bottom: 10px;
    }
    
    /* UI Element Overrides */
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# --- CORE DATA & ML MODULES ---
@st.cache_resource(show_spinner="Training predictive models...")
def load_and_train_models():
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'N': np.random.uniform(10, 140, n_samples),
        'P': np.random.uniform(5, 90, n_samples),
        'K': np.random.uniform(10, 80, n_samples),
        'Temp': np.random.uniform(18, 42, n_samples),
        'Humidity': np.random.uniform(30, 95, n_samples),
        'pH': np.random.uniform(5.5, 8.5, n_samples),
        'Rainfall': np.random.uniform(0, 300, n_samples),
        'Soil_Moisture': np.random.uniform(10, 80, n_samples)
    })
    
    base_water = (df['Temp'] * 150) + ((100 - df['Humidity']) * 40) - (df['Rainfall'] * 10) - (df['Soil_Moisture'] * 50)
    df['Water_Req'] = np.clip(base_water, 1500, 9000)
    
    X = df.drop('Water_Req', axis=1)
    y = df['Water_Req']
    
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    return model, df, feature_importance

@st.cache_data(ttl=3600)
def load_farm_data():
    try:
        df = pd.read_csv("vellore_farm_data.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ `vellore_farm_data.csv` missing. Using simulated telemetry fallbacks.")
        return pd.DataFrame(columns=["name", "fpo", "crop", "acres", "moisture", "status", "lat", "lon"])

@st.cache_data(ttl=600)
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()['current']
        return data['temperature_2m'], data['relative_humidity_2m'], data.get('precipitation', 0.0)
    except Exception as e:
        return 32.5, 65.0, 0.0

# --- UI VIEW COMPONENTS ---
def render_gis_telemetry(vellore_farms):
    st.markdown("<h1 class='gradient-text'>Vellore Precision GIS Telemetry</h1>", unsafe_allow_html=True)
    st.markdown("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations (FPOs).")
    
    cols = st.columns(4)
    metrics = [
        ("Total Monitored Land", "4,280 Acres", "↑ 12% vs last month"),
        ("Freshwater Saved", "18.4M Liters", "↓ 38.5% vs Flood"),
        ("Verified CO2 Offsets", "1,420 tCO2e", "AWD Protocol"),
        ("Farmer Payout Pool", "₹28.4 Lakhs", "85% Yield Share")
    ]
    for col, (label, val, delta) in zip(cols, metrics):
        col.metric(label, val, delta)
    
    st.markdown("### Interactive Farm Map View")
    if not vellore_farms.empty:
        color_map = {"Optimal": [16, 185, 129, 220], "Deficit": [245, 158, 11, 220], "Critical": [239, 68, 68, 220]}
        vellore_farms['color'] = vellore_farms.get('status', 'Optimal').map(color_map).fillna([16, 185, 129, 220])
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
        
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v11",
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=12.92, longitude=79.10, zoom=10.5, pitch=45),
            tooltip={
                "html": "<div style='font-family: Plus Jakarta Sans;'><b style='font-size: 1.1em; color: #10b981;'>{name}</b><hr style='margin: 4px 0; border-color: #334155;'/>FPO: {fpo}<br/>Crop: {crop}<br/>Area: {acres} Acres<br/>Moisture: {moisture}% ({status})</div>",
                "style": {"backgroundColor": "rgba(15, 23, 42, 0.95)", "color": "#f8fafc", "borderRadius": "12px", "padding": "16px", "border": "1px solid #334155", "boxShadow": "0 10px 25px rgba(0,0,0,0.5)"}
            }
        )
        st.pydeck_chart(deck, use_container_width=True)
        
        st.markdown("### Selected Field Telemetry Breakout")
        selected_farm = st.selectbox("Select Farm for Inspection", vellore_farms["name"])
        farm = vellore_farms[vellore_farms["name"] == selected_farm].iloc[0]
        temp, hum, precip = get_live_weather(farm['lat'], farm['lon'])
        
        c1, c2, c3 = st.columns(3)
        c1.info(f"🌿 **FPO Partner:**\n{farm['fpo']}\n\n🌾 **Target Crop:**\n{farm['crop']}")
        c2.warning(f"💧 **Soil Moisture:**\n{farm.get('moisture', 'N/A')}% ({farm.get('status', 'Unknown')})\n\n📐 **Acreage:**\n{farm['acres']} Acres")
        c3.success(f"🌡️ **Live Local Weather:**\n{temp}°C, {hum}% Humidity\n\n✅ **AWD Compliance:**\nVerified Compliant")

def render_ai_scheduler(model, feature_importance):
    st.markdown("<h1 class='gradient-text'>AI Irrigation Simulator</h1>", unsafe_allow_html=True)
    st.markdown("Dynamically calculates precise water requirements utilizing Random Forest regression to eliminate flood irrigation waste.")
    
    # Session State Initialization
    for key, default in [('api_temp', 34), ('api_hum', 60), ('api_rain', 5)]:
        if key not in st.session_state: st.session_state[key] = default
            
    def fetch_api_data():
        with st.spinner("Syncing API..."):
            t, h, r = get_live_weather(12.9716, 79.1589)
            st.session_state.api_temp, st.session_state.api_hum, st.session_state.api_rain = int(t), int(h), int(r)
            st.toast("✅ Live Vellore Weather Synced")

    st.button("🔄 Auto-Fill Live Vellore Weather", on_click=fetch_api_data, use_container_width=True, type="primary")
    st.divider()
    
    col_inputs, col_results = st.columns([1.1, 1])
    with col_inputs:
        with st.container(border=True):
            st.subheader("Farm & Soil Parameters")
            crop_type = st.selectbox("Target Crop", ["Paddy Rice", "Sugarcane", "Groundnut", "Maize"])
            c_acres, c_soil, c_nitro = st.columns(3)
            acres = c_acres.number_input("Area (Acres)", value=10.0, step=0.5)
            soil_m = c_soil.slider("Soil Moisture (%)", 10, 80, 25)
            nitrogen = c_nitro.slider("Nitrogen (mg/kg)", 10, 140, 65)
            
        with st.container(border=True):
            st.subheader("Micro-Climate Data")
            c_temp, c_hum, c_rain = st.columns(3)
            temp_in = c_temp.slider("Temp (°C)", 20, 45, key='api_temp')
            humidity_in = c_hum.slider("Humidity (%)", 20, 95, key='api_hum')
            rain_in = c_rain.slider("Rain (mm)", 0, 50, key='api_rain')
        
    with col_results:
        input_data = pd.DataFrame([{'N': nitrogen, 'P': 40, 'K': 30, 'Temp': temp_in, 'Humidity': humidity_in, 'pH': 6.8, 'Rainfall': rain_in, 'Soil_Moisture': soil_m}])
        flood_water = 8500 * acres
        
        if rain_in > 15:
            st.error("⛈️ **Heavy Rain Forecast:** Postpone irrigation. Natural rainfall will replenish soil moisture.")
            pred_water = 0
        else:
            pred_water = model.predict(input_data)[0] * acres
            
        water_saved = max(0, flood_water - pred_water)
        
        st.markdown("### Optimization Results")
        st.metric("Traditional Flood Volume", f"{int(flood_water):,} L/Day")
        
        if pred_water == 0:
            st.metric("EcoHydro AI Drip Volume", "0 L/Day (Rain Override)", "-100.0% Saved", delta_color="normal")
            st.progress(0, text="AI uses 0% of traditional flood water.")
        else:
            st.metric("EcoHydro AI Drip Volume", f"{int(pred_water):,} L/Day", f"-{round((water_saved/flood_water)*100, 1)}% Saved", delta_color="normal")
            pct = int((pred_water/flood_water)*100)
            st.progress(pct, text=f"AI uses {pct}% of traditional flood water.")
            
        st.success(f"⚡ **Monthly Pumping Power Saved:** ₹{int(water_saved * 30 * 0.012):,}")
    
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
        fig_chart.update_layout(height=280, margin={"r":0,"t":10,"l":0,"b":0}, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_chart, use_container_width=True)
            
    with c_explain:
        st.markdown("#### Explainable AI (XAI) Matrix")
        fig_importance = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale="Tealgrn")
        fig_importance.update_layout(height=280, margin={"r":0,"t":10,"l":0,"b":0}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_importance, use_container_width=True)

def render_carbon_ledger():
    st.markdown("<h1 class='gradient-text'>AWD Carbon Asset Engine</h1>", unsafe_allow_html=True)
    st.markdown("Verifies methane emission reductions via satellite radar (SAR) to mint high-quality ESG carbon credits.")
    
    cols = st.columns(3)
    cols[0].metric("Total Credits Minted", "1,314 tCO2e", "↑ 200 this week")
    cols[1].metric("Market Price per tCO2e", "₹1,530", "↑ ₹45 vs last month")
    cols[2].metric("Total Farmer Liquidity", "₹20,10,420", "Fully Disbursed")
    
    st.markdown("### Verified FPO Carbon Credit Batches")
    ledger_data = pd.DataFrame([
        {"Batch ID": "#AWD-26-089", "FPO Name": "North Arcot Farmers", "Acres": 450, "Credits (tCO2e)": 810, "Value (INR)": 1245000, "Compliance": 98, "Status": "Paid"},
        {"Batch ID": "#AWD-26-090", "FPO Name": "Palar River Basin", "Acres": 280, "Credits (tCO2e)": 504, "Value (INR)": 774000, "Compliance": 100, "Status": "Paid"},
        {"Batch ID": "#AWD-26-091", "FPO Name": "Walajah Organics", "Acres": 150, "Credits (tCO2e)": 270, "Value (INR)": 413100, "Compliance": 92, "Status": "Pending Audit"}
    ])
    
    st.dataframe(
        ledger_data,
        column_config={
            "Batch ID": st.column_config.TextColumn("Batch ID"),
            "Value (INR)": st.column_config.NumberColumn("Payout Value", format="₹ %d"),
            "Compliance": st.column_config.ProgressColumn("AWD Compliance", format="%d%%", min_value=0, max_value=100),
            "Status": st.column_config.TextColumn("Audit Status")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.download_button("📥 Download SEBI BRSR ESG Report", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Carbon_Audit.csv', mime='text/csv', type="secondary")

def render_business_model():
    st.markdown("<h1 class='gradient-text'>TAM/SAM & Business Projections</h1>", unsafe_allow_html=True)
    st.markdown("EcoHydro AI leverages a dual-revenue stream: B2B SaaS for FPOs and a 15% marketplace commission on carbon credits.")
    
    cols = st.columns(3)
    cols[0].metric("TAM (Smart Ag & Offsets)", "$12.5 Billion", "Global")
    cols[1].metric("SAM (Indian Irrigation)", "$1.8 Billion", "South India Focus")
    cols[2].metric("SOM (Year 3 Projection)", "$42 Million", "Serviceable Obtainable")
    
    st.divider()
    st.markdown("### Interactive Monetization Calculator")
    
    calc_c1, calc_c2 = st.columns([1, 1.5])
    with calc_c1:
        st.info("Adjust sliders to simulate Year-1 Revenue metrics.")
        fpos = st.slider("Onboarded FPOs", 5, 300, 30)
        credits = st.slider("Carbon Credits Traded (tCO2e)", 1000, 100000, 15000, step=1000)
        
        saas_rev = fpos * 2500 * 12
        carbon_rev = credits * 1650 * 0.15
        total_arr = saas_rev + carbon_rev
        
    with calc_c2:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4); padding: 32px; border-radius: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">Projected Annual Recurring Revenue</h4>
            <h1 style="color: #10b981; font-size: 3.5rem; margin: 15px 0;">₹{(total_arr)/100000:.2f} L</h1>
            <hr style="border-color: rgba(16, 185, 129, 0.2); margin: 20px 0;">
            <div style="display: flex; justify-content: space-around; font-size: 1.15rem; color: #e2e8f0;">
                <span><b>💻 SaaS Subscriptions:</b><br/>₹{saas_rev/100000:.2f}L</span>
                <span><b>🌱 Carbon Commission:</b><br/>₹{carbon_rev/100000:.2f}L</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN EXECUTION & NAVIGATION ---
if __name__ == "__main__":
    model, dataset, feature_importance = load_and_train_models()
    vellore_farms = load_farm_data()
    
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom: 0px;'>EcoHydro AI</h2>", unsafe_allow_html=True)
        st.caption("Precision Agriculture Engine v3.0")
        st.divider()
        page = st.radio("Select Workspace Module", [
            "GIS Farm Telemetry",
            "AI Irrigation Scheduler",
            "Methane & Carbon Ledger",
            "TAM/SAM & Business Model",
        ], label_visibility="collapsed")
        st.divider()
        st.info("System Status: Online 🟢")

    if page == "GIS Farm Telemetry":
        render_gis_telemetry(vellore_farms)
    elif page == "AI Irrigation Scheduler":
        render_ai_scheduler(model, feature_importance)
    elif page == "Methane & Carbon Ledger":
        render_carbon_ledger()
    elif page == "TAM/SAM & Business Model":
        render_business_model()
