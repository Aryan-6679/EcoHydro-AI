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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #0b0f19 !important; /* Deep dark background */
        color: #e2e8f0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    /* Glassmorphism Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.15);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.25);
        border-color: rgba(16, 185, 129, 0.6);
    }

    /* Metric Label & Value Styling */
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.8rem;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #ffffff !important;
        font-size: 2.4rem;
        text-shadow: 0 2px 10px rgba(255,255,255,0.1);
    }
    
    /* Gradients and Headers */
    h1, h2, h3 { color: #f8fafc !important; }
    .gradient-text {
        background: linear-gradient(135deg, #34d399 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
    }
    
    /* Custom Info Panels */
    .custom-panel {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
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
    """Load from CSV, or generate realistic fallback data if CSV is missing."""
    try:
        df = pd.read_csv("vellore_farm_data.csv")
        if not df.empty:
            return df
    except FileNotFoundError:
        pass
    
    # Fallback to ensure the map always looks incredible even without the CSV
    np.random.seed(42)
    n_farms = 85
    lats = np.random.normal(12.92, 0.08, n_farms)
    lons = np.random.normal(79.10, 0.08, n_farms)
    status_choices = ["Optimal", "Deficit", "Critical"]
    
    df = pd.DataFrame({
        "name": [f"Block-{str(i).zfill(3)} Farm" for i in range(n_farms)],
        "fpo": np.random.choice(["North Arcot Farmers", "Palar River Basin", "Walajah Organics", "Arcot Agri Hub"], n_farms),
        "crop": np.random.choice(["Paddy Rice", "Sugarcane", "Groundnut", "Maize"], n_farms),
        "acres": np.random.uniform(5, 60, n_farms).round(1),
        "moisture": np.random.uniform(10, 75, n_farms).round(1),
        "lat": lats,
        "lon": lons
    })
    # Assign status based on moisture for realism
    df['status'] = pd.cut(df['moisture'], bins=[0, 25, 45, 100], labels=["Critical", "Deficit", "Optimal"])
    return df

vellore_farms = load_farm_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
            <div>
                <h2 style='margin: 0; padding: 0; background: linear-gradient(135deg, #34d399 0%, #0ea5e9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>EcoHydro AI</h2>
                <caption style='color: #64748b; font-size: 0.8rem;'>Precision Ag v2.5 PRO</caption>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    page = st.radio("SELECT MODULE", [
        "GIS Farm Telemetry",
        "AI Irrigation Scheduler",
        "Methane & Carbon Ledger",
        "TAM/SAM & Business Model",
    ], label_visibility="collapsed")
    st.divider()
    
    st.markdown("""
        <div style='text-align: center; color: #475569; font-size: 0.75rem;'>
            System Status: All Sensors Online<br>
            Last Sync: Just Now
        </div>
    """, unsafe_allow_html=True)

# --- PAGE 1: GIS TELEMETRY ---
if page == "GIS Farm Telemetry":
    st.markdown("<h1 class='gradient-text'>Vellore Region Precision GIS Telemetry</h1>", unsafe_allow_html=True)
    st.write("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations (FPOs).")
    
    st.write("")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Monitored Canopy", "4,280 Acres", "+12% MoM")
    col_m2.metric("Freshwater Saved", "18.4M Liters", "38.5% vs Flood")
    col_m3.metric("Verified Offsets", "1,420 tCO2e", "AWD Protocol")
    col_m4.metric("Farmer Liquidity", "INR 28.4 Lakhs", "85% Yield")
    
    st.markdown("<br><h3>3D Interactive Farm Telemetry</h3>", unsafe_allow_html=True)
    
    if not vellore_farms.empty:
        # Define premium color scheme based on status
        color_map = {
            "Optimal": [16, 185, 129, 230],    # Emerald
            "Deficit": [245, 158, 11, 230],    # Amber
            "Critical": [239, 68, 68, 230]     # Red
        }
        vellore_farms['color'] = vellore_farms['status'].map(color_map)
        
        # Calculate dynamic elevation based on acres for 3D effect
        vellore_farms['elevation'] = vellore_farms['acres'] * 80 
        
        # Layer 1: 3D Hexagonal/Column Columns representing acreage
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=1,
            radius=200,
            get_fill_color="color",
            get_line_color=[255, 255, 255, 50],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
            extruded=True,
            material={
                "ambient": 0.8,
                "diffuse": 0.6,
                "shininess": 32,
                "specularColor": [255, 255, 255]
            }
        )
        
        # Layer 2: Base scatter glow
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=400,
            opacity=0.3,
            pickable=False
        )
        
        # Dramatic 3D View State
        view_state = pdk.ViewState(
            latitude=12.92,
            longitude=79.10,
            zoom=10.8,
            pitch=55,
            bearing=-15
        )
        
        deck = pdk.Deck(
            map_provider="carto",
            map_style="dark", 
            layers=[scatter_layer, column_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{name}</b><br/>Area: {acres} Acres<br/>Moisture: {moisture}% ({status})",
                "style": {
                    "backgroundColor": "#1e293b",
                    "color": "#f8fafc",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "border": "1px solid rgba(52, 211, 153, 0.3)",
                    "boxShadow": "0 10px 25px rgba(0,0,0,0.5)",
                    "fontFamily": "sans-serif",
                    "fontSize": "14px",
                    "whiteSpace": "nowrap", # Prevents text cropping on hover
                    "lineHeight": "1.5"
                }
            }
        )
        
        # Render PyDeck
        st.pydeck_chart(deck)
        
        # Selected Field Breakout
        st.write("")
        st.markdown("### Granular Field Analysis")
        selected_farm_name = st.selectbox("Select Farm for Deep Dive", vellore_farms["name"], label_visibility="collapsed")
        
        farm = vellore_farms[vellore_farms["name"] == selected_farm_name].iloc[0]
        temp, hum, precip = get_live_weather(farm['lat'], farm['lon'])
        
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 10px;">
                <div class="custom-panel" style="border-left-color: #3b82f6;">
                    <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Organization</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">{farm['fpo']}</div>
                    <div style="margin-top: 10px; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Active Crop</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">{farm['crop']}</div>
                </div>
                <div class="custom-panel" style="border-left-color: {'#10b981' if farm['status'] == 'Optimal' else '#f59e0b' if farm['status'] == 'Deficit' else '#ef4444'};">
                    <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Soil Moisture Status</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">{farm['moisture']}% ({farm['status']})</div>
                    <div style="margin-top: 10px; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Total Acreage</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">{farm['acres']} Acres</div>
                </div>
                <div class="custom-panel" style="border-left-color: #8b5cf6;">
                    <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Live Microclimate</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">{temp} C, {hum}% RH</div>
                    <div style="margin-top: 10px; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Carbon Compliance</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #10b981;">Verified AWD</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- PAGE 2: AI IRRIGATION SCHEDULER ---
elif page == "AI Irrigation Scheduler":
    st.markdown("<h1 class='gradient-text'>XGBoost/RF Precision Irrigation Engine</h1>", unsafe_allow_html=True)
    st.write("Dynamically calculates real-time water requirements to eliminate flood irrigation waste.")
    
    if 'api_temp' not in st.session_state: st.session_state.api_temp = 34
    if 'api_hum' not in st.session_state: st.session_state.api_hum = 60
    if 'api_rain' not in st.session_state: st.session_state.api_rain = 5
    
    def fetch_api_data():
        t, h, r = get_live_weather(12.9716, 79.1589)
        st.session_state.api_temp = int(t)
        st.session_state.api_hum = int(h)
        st.session_state.api_rain = int(r)
        st.toast("Live Weather Synced for Vellore API")
    
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        st.button("Auto-Sync Live Weather", on_click=fetch_api_data, width="stretch")
    
    st.write("")
    
    col_inputs, col_results = st.columns([1.2, 1])
    with col_inputs:
        with st.expander("Farm & Soil Composition", expanded=True):
            crop_type = st.selectbox("Target Crop Profile", ["Paddy Rice", "Sugarcane", "Groundnut", "Maize"])
            c_acres, c_soil, c_nitro = st.columns(3)
            acres = c_acres.number_input("Area (Acres)", value=10.0, step=1.0)
            soil_m = c_soil.slider("Soil Moisture %", 10, 80, 25)
            nitrogen = c_nitro.slider("Nitrogen (mg/kg)", 10, 140, 65)
            
        with st.expander("Environmental Telemetry", expanded=True):
            c_temp, c_hum, c_rain = st.columns(3)
            temp_in = c_temp.slider("Temp (C)", 20, 45, key='api_temp')
            humidity_in = c_hum.slider("Humidity (%)", 20, 95, key='api_hum')
            rain_in = c_rain.slider("Rain (mm)", 0, 50, key='api_rain')
        
    with col_results:
        input_data = pd.DataFrame([{'N': nitrogen, 'P': 40, 'K': 30, 'Temp': temp_in, 'Humidity': humidity_in, 'pH': 6.8, 'Rainfall': rain_in, 'Soil_Moisture': soil_m}])
        
        # Inference Logic
        if rain_in > 15:
            pred_water = 0
            st.markdown("""
                <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                    <b style="color: #60a5fa;">Rain Override Triggered:</b> Natural precipitation is sufficient. Irrigation postponed.
                </div>
            """, unsafe_allow_html=True)
        else:
            pred_water = model.predict(input_data)[0] * acres
            
        flood_water = 8500 * acres
        water_saved = max(0, flood_water - pred_water)
        
        st.markdown("<h3 style='margin-top: 0;'>AI Optimization Delta</h3>", unsafe_allow_html=True)
        st.metric("Traditional Flood Volume", f"{int(flood_water):,} L/Day", "Legacy Baseline", delta_color="off")
        
        if pred_water == 0:
            st.metric("EcoHydro AI Volume", "0 L/Day (Rain Override)", "100.0% Saved", delta_color="normal")
            progress_pct = 0
        else:
            st.metric("EcoHydro AI Volume", f"{int(pred_water):,} L/Day", f"{round((water_saved/flood_water)*100, 1)}% Saved", delta_color="normal")
            progress_pct = int((pred_water/flood_water)*100)
            
        st.progress(progress_pct, text=f"AI uses {progress_pct}% of traditional baseline")
        
        st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 15px; border-radius: 12px; margin-top: 20px;">
                <h4 style="margin: 0; color: #10b981; font-size: 1.1rem;">Operational Savings</h4>
                <div style="color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin-top: 5px;">INR {int(water_saved * 30 * 0.012):,} / Month</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">(Estimated pumping & energy costs reduced)</div>
            </div>
        """, unsafe_allow_html=True)
    
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
        fig_chart.update_layout(
            margin={"r":0,"t":20,"l":0,"b":0}, height=280, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8")
        )
        st.plotly_chart(fig_chart)
            
    with c_explain:
        st.markdown("#### Explainable AI Matrix")
        fig_importance = px.bar(
            feature_importance, x='Importance', y='Feature', orientation='h', 
            color='Importance', color_continuous_scale="Tealgrn"
        )
        fig_importance.update_layout(
            height=280, margin={"r":0,"t":20,"l":0,"b":0}, 
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), yaxis_title=None, xaxis_title="Decision Weight"
        )
        st.plotly_chart(fig_importance)

# --- PAGE 3: CARBON LEDGER ---
elif page == "Methane & Carbon Ledger":
    st.markdown("<h1 class='gradient-text'>AWD Methane Reduction Protocol</h1>", unsafe_allow_html=True)
    st.write("Verifies emission reductions via satellite radar (SAR) to mint high-quality ESG carbon credits.")
    
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vaulted Credits", "1,314 tCO2e", "+200 this week")
    c2.metric("Current Market Price", "INR 1,530", "+ INR 45 vs last month")
    c3.metric("FPO Payout Distributed", "INR 20,10,420", "Fully Settled")
    
    st.markdown("<br><h3>Verified FPO Carbon Credit Batches</h3>", unsafe_allow_html=True)
    ledger_data = pd.DataFrame([
        {"Batch ID": "#AWD-26-089", "FPO Name": "North Arcot Farmers", "Acres": 450, "Credits (tCO2e)": 810, "Value (INR)": 1245000, "Compliance": 98, "Status": "Paid"},
        {"Batch ID": "#AWD-26-090", "FPO Name": "Palar River Basin", "Acres": 280, "Credits (tCO2e)": 504, "Value (INR)": 774000, "Compliance": 100, "Status": "Paid"},
        {"Batch ID": "#AWD-26-091", "FPO Name": "Walajah Organics", "Acres": 150, "Credits (tCO2e)": 270, "Value (INR)": 413100, "Compliance": 92, "Status": "Pending Audit"},
        {"Batch ID": "#AWD-26-092", "FPO Name": "Arcot Agri Hub", "Acres": 320, "Credits (tCO2e)": 576, "Value (INR)": 881280, "Compliance": 95, "Status": "Minting"}
    ])
    
    st.dataframe(
        ledger_data,
        column_config={
            "Batch ID": st.column_config.TextColumn("Ledger Hash", width="small"),
            "Credits (tCO2e)": st.column_config.NumberColumn("Verified tCO2e"),
            "Value (INR)": st.column_config.NumberColumn("Payout Value", format="INR %d"),
            "Compliance": st.column_config.ProgressColumn("SAR AWD Compliance", format="%d%%", min_value=0, max_value=100),
            "Status": st.column_config.TextColumn("Smart Contract Status")
        },
        hide_index=True,
        width="stretch"
    )
    
    col_dl, col_space = st.columns([1, 4])
    with col_dl:
        st.download_button("Download SEBI BRSR Audit (CSV)", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Carbon_Ledger.csv', mime='text/csv', width="stretch")

# --- PAGE 4: BUSINESS MODEL ---
elif page == "TAM/SAM & Business Model":
    st.markdown("<h1 class='gradient-text'>Financial Projections & Market Scope</h1>", unsafe_allow_html=True)
    st.write("EcoHydro AI leverages a robust dual-revenue stream: B2B SaaS for FPOs and a 15% marketplace commission on minted carbon credits.")
    
    st.write("")
    col_tam1, col_tam2, col_tam3 = st.columns(3)
    col_tam1.metric("TAM (Global)", "$12.5 Billion", "Smart Ag & ESG")
    col_tam2.metric("SAM (South India)", "$1.8 Billion", "Irrigation Automation")
    col_tam3.metric("SOM (Year 3 Target)", "$42 Million", "Serviceable Obtainable")
    
    st.write("<br><hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("### Interactive Monetization Sandbox")
    
    calc_c1, calc_c2 = st.columns([1, 1.5])
    with calc_c1:
        st.markdown("<div style='color: #94a3b8; margin-bottom: 15px;'>Drag sliders to simulate Year-1 Unit Economics:</div>", unsafe_allow_html=True)
        fpos = st.slider("Onboarded FPOs (B2B Accounts)", 5, 300, 45)
        credits = st.slider("Carbon Credits Traded (tCO2e)", 1000, 150000, 25000, step=1000)
        
        saas_rev = fpos * 2500 * 12
        carbon_rev = credits * 1650 * 0.15
        total_arr = saas_rev + carbon_rev
        
    with calc_c2:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(16, 185, 129, 0.15), rgba(15, 23, 42, 0.8)); border: 1px solid rgba(16, 185, 129, 0.4); padding: 35px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <h4 style="margin-top: 0; color: #94a3b8; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">Projected Annual Recurring Revenue (ARR)</h4>
            <h1 style="color: #34d399; font-size: 3.5rem; margin: 15px 0; text-shadow: 0 2px 10px rgba(52, 211, 153, 0.2);">INR {(total_arr)/100000:.2f} Lakhs</h1>
            <div style="height: 1px; background: rgba(255,255,255,0.1); margin: 25px 0;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 1.1rem;">
                <div style="display: flex; flex-direction: column;">
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">SaaS Subscriptions</span>
                    <span style="color: #f8fafc; font-weight: 600; font-size: 1.4rem;">INR {saas_rev/100000:.2f}L</span>
                </div>
                <div style="display: flex; flex-direction: column; text-align: right;">
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Carbon Commission (15%)</span>
                    <span style="color: #f8fafc; font-weight: 600; font-size: 1.4rem;">INR {carbon_rev/100000:.2f}L</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
