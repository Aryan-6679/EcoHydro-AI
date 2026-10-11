import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import requests
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="EcoHydro AI - Precision Agriculture",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Premium Glassmorphism Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.8), rgba(31, 41, 55, 0.4));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.25);
        border-color: rgba(16, 185, 129, 0.6);
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 55, 0.6));
    }

    /* Metric Label & Value Styling */
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #ffffff !important;
        font-size: 2.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Beautiful Gradients for Headers */
    h1, h2, h3 { color: #f9fafb !important; }
    .gradient-text {
        background: linear-gradient(135deg, #34d399 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Customizing Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(31, 41, 55, 0.5) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 2rem 0;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_models():
    """Generates synthetic data and trains a RandomForest for water requirement prediction."""
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

def load_farm_data():
    """Robustly load and sanitize farm data, strictly removing duplicate columns."""
    df = pd.DataFrame()
    try:
        df = pd.read_csv("vellore_farm_data.csv")
        
        # 1. Clean column strings completely
        df.columns = df.columns.astype(str).str.strip().str.lower()
        
        # 2. CRITICAL FIX: Drop any duplicate columns before renaming (prevents the pandas/PyDeck crash)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 3. Rename custom CSV headers to standard app variables
        col_mapping = {
            'farm': 'name', 'farm name': 'name', 'farm_name': 'name', 'id': 'name',
            'farmer': 'name', 'farmer name': 'name', 'plot': 'name', 'location': 'name', 'village': 'name',
            'area': 'acres', 'area (acres)': 'acres', 'size': 'acres', 'acreage': 'acres',
            'soil moisture': 'moisture', 'soil_moisture': 'moisture', 'humidity': 'moisture',
            'fpo name': 'fpo', 'organization': 'fpo', 'cluster': 'fpo'
        }
        df.rename(columns=col_mapping, inplace=True)
        
        # 4. CRITICAL FIX: Drop duplicates again just in case the mapping created clashes
        df = df.loc[:, ~df.columns.duplicated()]
        
    except FileNotFoundError:
        pass
    
    # 5. Generate beautiful 3D fallback data if CSV is empty or missing
    if df.empty:
        np.random.seed(42)
        num_farms = 25
        center_lat, center_lon = 12.92, 79.10
        
        vellore_regions = ["Karnampattu", "Gudiyatham", "Pallikondai", "Katpadi", "Anaicut", "Arcot", "Walajah", "Ranipet"]
        
        df = pd.DataFrame({
            "name": [f"{vellore_regions[i % len(vellore_regions)]} Plot {i+1}" for i in range(num_farms)],
            "lat": center_lat + np.random.uniform(-0.15, 0.15, num_farms),
            "lon": center_lon + np.random.uniform(-0.15, 0.15, num_farms),
            "moisture": np.random.uniform(15, 60, num_farms).round(1),
            "acres": np.random.uniform(5, 45, num_farms).round(1),
            "fpo": np.random.choice(["North Arcot Farmers", "Palar River Basin", "Walajah Organics"], num_farms),
            "crop": np.random.choice(["Paddy Rice", "Sugarcane", "Groundnut"], num_farms)
        })
        
    # 6. GUARANTEE all required columns exist and are safely formatted for PyDeck HTML injection
    if 'name' not in df.columns: 
        local_regions = ["Karnampattu", "Gudiyatham", "Pallikondai", "Katpadi", "Anaicut", "Walajah"]
        df['name'] = [f"{local_regions[i % len(local_regions)]} Field {i+1}" for i in range(len(df))]
    if 'fpo' not in df.columns: df['fpo'] = "Vellore Agri Hub"
    if 'crop' not in df.columns: df['crop'] = "Mixed Crop"
    if 'acres' not in df.columns: df['acres'] = 10.0
    if 'moisture' not in df.columns: df['moisture'] = 45.0
    if 'lat' not in df.columns: df['lat'] = 12.92
    if 'lon' not in df.columns: df['lon'] = 79.10
    
    df['name'] = df['name'].fillna('Unknown').astype(str)
    df['fpo'] = df['fpo'].fillna('Unknown').astype(str)
    df['crop'] = df['crop'].fillna('Unknown').astype(str)
    df['acres'] = pd.to_numeric(df['acres'], errors='coerce').fillna(10.0).round(1)
    df['moisture'] = pd.to_numeric(df['moisture'], errors='coerce').fillna(45.0).round(1)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce').fillna(12.92)
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce').fillna(79.10)
    
    return df

vellore_farms = load_farm_data()

vellore_farms['status'] = pd.cut(
    vellore_farms['moisture'], 
    bins=[0, 25, 45, 100], 
    labels=["Critical", "Deficit", "Optimal"]
).astype(str)

with st.sidebar:
    st.markdown("<h1 style='margin-top: -10px; margin-bottom: 0px; font-weight: 800; color: #10b981 !important; letter-spacing: -1px;'>EcoHydro AI</h1>", unsafe_allow_html=True)
    st.caption("PRECISION AGRICULTURE OS v3.0")
    
    st.write("")
    st.write("")
    page = st.radio("WORKSPACE MODULES", [
        "GIS Farm Telemetry",
        "AI Irrigation Scheduler",
        "Methane & Carbon Ledger",
        "TAM/SAM & Business Model",
    ], label_visibility="collapsed")
    
    st.write("")
    st.write("")
    st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 10px;">
            <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase;">System Status</div>
            <div style="color: #10b981; font-weight: 600; margin-top: 5px;">All Sensors Online</div>
            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 5px;">Last Sync: Just Now</div>
        </div>
    """, unsafe_allow_html=True)

if page == "GIS Farm Telemetry":
    st.markdown("<h1 class='gradient-text'>3D Interactive Farm Telemetry</h1>", unsafe_allow_html=True)
    st.write("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations.")
    
    st.write("")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Monitored Land", "4,280 Acres", "+12% this month")
    col_m2.metric("Freshwater Saved", "18.4M Liters", "38.5% vs Flood")
    col_m3.metric("Verified CO2 Offsets", "1,420 tCO2e", "AWD Method")
    col_m4.metric("Farmer Payout Pool", "INR 28.4 Lakhs", "85% Yield")
    
    st.write("")
    if not vellore_farms.empty:
        color_map = {
            "Optimal": [16, 185, 129, 255],     # Green
            "Deficit": [245, 158, 11, 255],     # Yellow
            "Critical": [239, 68, 68, 255]      # Red
        }
        vellore_farms['color'] = vellore_farms['status'].map(color_map)
        
        # Glow Effect Layer (Soft halo around the base)
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="acres * 80",
            pickable=False,
            opacity=0.3
        )
        
        # 3D Column Layer
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_elevation="acres",
            elevation_scale=150,
            radius=350,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True
        )
        
        view_state = pdk.ViewState(
            latitude=vellore_farms['lat'].mean(),
            longitude=vellore_farms['lon'].mean(),
            zoom=10.2,
            pitch=55,
            bearing=-15
        )
        
        # Bulletproof Tooltip Structure (No Tables, CSS Line breaks, forced nowrap)
        tooltip_html = (
            "<b>Farm:</b> {name} <br/>"
            "<b>FPO Partner:</b> {fpo} <br/>"
            "<b>Harvest Crop:</b> {crop} <br/>"
            "<b>Field Area:</b> {acres} Acres <br/>"
            "<b>Soil Moisture:</b> {moisture}% ({status})"
        )
        
        deck = pdk.Deck(
            map_provider="carto",
            map_style="dark", 
            layers=[scatter_layer, column_layer],
            initial_view_state=view_state,
            tooltip={
                "html": tooltip_html,
                "style": {
                    "backgroundColor": "rgba(17, 24, 39, 0.95)",
                    "color": "#f3f4f6",
                    "border": "1px solid #10b981",
                    "borderRadius": "12px",
                    "padding": "16px",
                    "boxShadow": "0 10px 40px rgba(0,0,0,0.8)",
                    "fontFamily": "Inter, sans-serif",
                    "fontSize": "14px",
                    "lineHeight": "1.7",
                    "zIndex": "99999",
                    "whiteSpace": "nowrap",
                    "backdropFilter": "blur(4px)"
                }
            }
        )
        
        st.pydeck_chart(deck, width="stretch")
        
        st.write("")
        st.markdown("### Granular HUD Analysis")
        selected_farm_name = st.selectbox("Select Farm for Deep Inspection", vellore_farms["name"].unique(), label_visibility="collapsed")
        
        farm_subset = vellore_farms[vellore_farms["name"] == selected_farm_name]
        
        if not farm_subset.empty:
            farm = farm_subset.iloc[0]
            temp, hum, precip = get_live_weather(farm['lat'], farm['lon'])
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="background: rgba(31, 41, 55, 0.6); padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                    <div style="color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;">FPO Partner</div>
                    <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">{farm['fpo']}</div>
                    <div style="color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;">Target Crop</div>
                    <div style="color: white; font-size: 1.1rem; font-weight: 600;">{farm['crop']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                status_color = "#10b981" if farm['moisture'] >= 30 else "#ef4444"
                st.markdown(f"""
                <div style="background: rgba(31, 41, 55, 0.6); padding: 20px; border-radius: 12px; border-left: 4px solid {status_color};">
                    <div style="color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;">Soil Moisture</div>
                    <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">{farm['moisture']}% ({'Optimal' if farm['moisture'] >= 30 else 'Deficit'})</div>
                    <div style="color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;">Acreage</div>
                    <div style="color: white; font-size: 1.1rem; font-weight: 600;">{farm['acres']} Acres</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background: rgba(31, 41, 55, 0.6); padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
                    <div style="color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;">Live Local Weather</div>
                    <div style="color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">{temp} C, {hum}% Humidity</div>
                    <div style="color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;">AWD Protocol Compliance</div>
                    <div style="color: #10b981; font-size: 1.1rem; font-weight: 600;">Verified Compliant</div>
                </div>
                """, unsafe_allow_html=True)

elif page == "AI Irrigation Scheduler":
    st.markdown("<h1 class='gradient-text'>AI Irrigation Simulator</h1>", unsafe_allow_html=True)
    st.write("Dynamically calculates precise water requirements utilizing Random Forest predictions to eliminate flood irrigation waste.")
    
    if 'api_temp' not in st.session_state: st.session_state.api_temp = 34
    if 'api_hum' not in st.session_state: st.session_state.api_hum = 60
    if 'api_rain' not in st.session_state: st.session_state.api_rain = 5
    
    def fetch_api_data():
        t, h, r = get_live_weather(12.9716, 79.1589)
        st.session_state.api_temp = int(t)
        st.session_state.api_hum = int(h)
        st.session_state.api_rain = int(r)
        st.toast("Live Vellore Weather Synced")
    
    st.button("Auto-Fill Live Vellore Weather", on_click=fetch_api_data)
    st.write("")
    
    col_inputs, col_results = st.columns([1.2, 1], gap="large")
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
        
        st.markdown("<h3 style='margin-top: 0; margin-bottom: 20px;'>Optimization Results</h3>", unsafe_allow_html=True)
        st.metric("Traditional Flood Volume", f"{int(flood_water):,} L/Day")
        
        if pred_water == 0:
            st.metric("EcoHydro AI Drip Volume", "0 L/Day (Rain Override)", "-100.0% Saved", delta_color="normal")
            progress_pct = 0
        else:
            st.metric("EcoHydro AI Drip Volume", f"{int(pred_water):,} L/Day", f"-{round((water_saved/flood_water)*100, 1)}% Saved", delta_color="normal")
            progress_pct = int((pred_water/flood_water)*100)
            
        st.progress(progress_pct, text=f"AI uses only {progress_pct}% of traditional flood water.")
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%); border: 1px solid rgba(16, 185, 129, 0.3); padding: 20px; border-radius: 16px; margin-top: 25px;">
            <div style="color: #10b981; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.85rem;">Monthly Pumping Cost Saved</div>
            <div style="font-size: 2rem; font-weight: 800; color: white; margin-top: 5px;">INR {int(water_saved * 30 * 0.012):,}</div>
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
        fig_chart.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=300, legend=dict(yanchor="bottom", y=-0.5, xanchor="center", x=0.5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af")
        st.plotly_chart(fig_chart, use_container_width=True)
            
    with c_explain:
        st.markdown("#### Explainable AI Feature Weights")
        fig_importance = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale="Tealgrn")
        fig_importance.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af")
        st.plotly_chart(fig_importance, use_container_width=True)

elif page == "Methane & Carbon Ledger":
    st.markdown("<h1 class='gradient-text'>AWD Carbon Protocol Engine</h1>", unsafe_allow_html=True)
    st.write("Verifies methane emission reductions via satellite radar (SAR) to mint high-quality ESG carbon credits.")
    
    st.write("")
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
    
    st.download_button("Download SEBI BRSR ESG Report (CSV)", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Carbon_Audit.csv', mime='text/csv')

elif page == "TAM/SAM & Business Model":
    st.markdown("<h1 class='gradient-text'>Financial Projections & Market Sizing</h1>", unsafe_allow_html=True)
    st.write("EcoHydro AI leverages a dual-revenue stream: B2B SaaS for FPOs and a 15% marketplace commission on carbon credits.")
    
    st.write("")
    col_tam1, col_tam2, col_tam3 = st.columns(3)
    col_tam1.metric("TAM (Smart Ag & Offsets)", "$12.5 Billion", "Global")
    col_tam2.metric("SAM (Indian Irrigation Automation)", "$1.8 Billion", "South India Focus")
    col_tam3.metric("SOM (Year 3 Projection)", "$42 Million", "Serviceable Obtainable")
    
    st.write("---")
    st.markdown("### Interactive Monetization Simulator")
    
    calc_c1, calc_c2 = st.columns([1, 1.5], gap="large")
    with calc_c1:
        st.info("Adjust the sliders below to simulate Year-1 Revenue")
        fpos = st.slider("Onboarded FPOs", 5, 300, 30)
        credits = st.slider("Carbon Credits Traded (tCO2e)", 1000, 100000, 15000, step=1000)
        
        saas_rev = fpos * 2500 * 12
        carbon_rev = credits * 1650 * 0.15
        total_arr = saas_rev + carbon_rev
        
    with calc_c2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%); border: 1px solid rgba(16, 185, 129, 0.4); padding: 40px; border-radius: 20px; box-shadow: 0 10px 40px rgba(16, 185, 129, 0.15);">
            <h4 style="margin-top: 0; color: #9ca3af; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Projected Annual Recurring Revenue</h4>
            <h1 style="color: #ffffff; font-size: 4rem; margin: 15px 0; font-weight: 800; text-shadow: 0 2px 15px rgba(16,185,129,0.4);">INR {(total_arr)/100000:.2f} Lakhs</h1>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 25px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 1.2rem; color: #e5e7eb;">
                <span><b>SaaS Subscriptions:</b><br/><span style="color:#10b981;">INR {saas_rev/100000:.2f}L / Year</span></span>
                <span style="text-align: right;"><b>Carbon Commission:</b><br/><span style="color:#06b6d4;">INR {carbon_rev/100000:.2f}L / Year</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
