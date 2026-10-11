import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import requests
import re
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION & PREMIUM UI ---
st.set_page_config(
    page_title="EcoHydro AI | Precision Agriculture OS",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #09090b !important; /* Zinc 950 - Deepest black */
        color: #e4e4e7; /* Zinc 200 */
    }
    
    /* Hide Streamlit dev elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111113 !important;
        border-right: 1px solid #27272a;
    }
    
    /* Metric Cards - Premium Glass & Gradient Style */
    [data-testid="metric-container"] {
        background: linear-gradient(180deg, #18181b 0%, #111113 100%);
        padding: 24px 20px;
        border-radius: 12px;
        border: 1px solid #27272a;
        border-top: 1px solid #3f3f46; /* Subtle light top border for 3D effect */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: #52525b;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #a1a1aa !important; /* Zinc 400 */
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    [data-testid="stMetricValue"] {
        color: #fafafa !important; /* Zinc 50 */
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #fafafa !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 1px solid #27272a;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    /* HUD Cards for Granular Data */
    .hud-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .hud-label {
        font-size: 0.75rem;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .hud-value {
        font-size: 1.15rem;
        color: #fafafa;
        font-weight: 500;
        margin-top: 2px;
    }
    
    /* Status Colors */
    .status-optimal { border-left: 4px solid #10b981; }
    .status-deficit { border-left: 4px solid #f59e0b; }
    .status-critical { border-left: 4px solid #ef4444; }
    .status-info { border-left: 4px solid #3b82f6; }
    
    /* Buttons */
    .stButton>button {
        background-color: #fafafa;
        color: #09090b;
        border: 1px solid #fafafa;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #e4e4e7;
        color: #09090b;
        border-color: #e4e4e7;
    }
    
    /* Expanders & Dividers */
    .streamlit-expanderHeader {
        background-color: #18181b !important;
        border-radius: 8px !important;
        border: 1px solid #27272a;
        font-weight: 500;
    }
    hr { border-color: #27272a !important; margin: 2rem 0; }
    
    /* Streamlit Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #27272a;
    }
    </style>
""", unsafe_allow_html=True)

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
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()['current']
            return data['temperature_2m'], data['relative_humidity_2m'], data.get('precipitation', 0.0)
    except Exception:
        pass
    return 32.5, 65.0, 0.0

def load_farm_data():
    df = pd.DataFrame()
    try:
        raw_df = pd.read_csv("vellore_farm_data.csv")
        
        # PRE-SANITIZATION: Strip exact duplicate columns instantly to prevent Pandas crashing
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
        
        # Standardize column headers to lowercase
        raw_df.columns = raw_df.columns.astype(str).str.strip().str.lower()
        
        col_mapping = {
            'farm': 'name', 'farm name': 'name', 'farm_name': 'name', 'id': 'name',
            'farmer': 'name', 'farmer name': 'name', 'plot': 'name', 'location': 'name', 'village': 'name',
            'area': 'acres', 'area (acres)': 'acres', 'size': 'acres', 'acreage': 'acres', 'area_acres': 'acres',
            'soil moisture': 'moisture', 'soil_moisture': 'moisture', 'humidity': 'moisture',
            'fpo name': 'fpo', 'organization': 'fpo', 'cluster': 'fpo',
            'crop name': 'crop', 'target crop': 'crop', 'harvest': 'crop'
        }
        df = raw_df.rename(columns=col_mapping)
        
        # POST-SANITIZATION: Ensure no duplicate columns were created during rename
        df = df.loc[:, ~df.columns.duplicated()]
        
    except Exception:
        pass

    # Fallback Generation if empty or completely failed
    if df.empty:
        np.random.seed(42)
        num_farms = 25
        df = pd.DataFrame({
            "name": [f"Farm {i}" for i in range(num_farms)],
            "lat": 12.92 + np.random.uniform(-0.15, 0.15, num_farms),
            "lon": 79.10 + np.random.uniform(-0.15, 0.15, num_farms),
            "moisture": np.random.uniform(15, 60, num_farms).round(1),
            "acres": np.random.uniform(5, 45, num_farms).round(1),
            "fpo": np.random.choice(["North Arcot Farmers", "Palar Basin Co.", "Walajah Organics"], num_farms),
            "crop": np.random.choice(["Paddy Rice", "Sugarcane", "Groundnut"], num_farms)
        })

    # Ensure required columns exist, mapping defaults if completely absent
    for col in ['name', 'fpo', 'crop', 'acres', 'moisture', 'lat', 'lon']:
        if col not in df.columns:
            if col == 'fpo': df[col] = "Vellore Agri Hub"
            elif col == 'crop': df[col] = "Mixed Crop"
            elif col == 'name': df[col] = "Unknown Tract"
            elif col == 'acres': df[col] = 10.0
            elif col == 'moisture': df[col] = 45.0
            elif col == 'lat': df[col] = 12.92
            elif col == 'lon': df[col] = 79.10

    # Intelligent Name Enhancement (replaces F-01 with realistic Vellore names)
    def enhance_name(val, idx):
        val_str = str(val).strip()
        if pd.isna(val) or re.match(r'^(F-?\d+|\d+|Farm\s*\d+|Plot\s*\d+)$', val_str, re.IGNORECASE):
            regions = ["Karnampattu", "Gudiyatham", "Pallikondai", "Katpadi", "Anaicut", "Walajah", "Ranipet"]
            return f"{regions[idx % len(regions)]} Sector {idx+1}"
        return val_str

    df['name'] = [enhance_name(x, i) for i, x in enumerate(df['name'])]

    # Enforce strict typing to prevent PyDeck/PyArrow serialization drops
    df['name'] = df['name'].astype(str)
    df['fpo'] = df['fpo'].astype(str)
    df['crop'] = df['crop'].astype(str)
    df['acres'] = pd.to_numeric(df['acres'], errors='coerce').fillna(10.0).round(1)
    df['moisture'] = pd.to_numeric(df['moisture'], errors='coerce').fillna(45.0).round(1)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce').fillna(12.92)
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce').fillna(79.10)

    # Add Categorical Status
    df['status'] = pd.cut(
        df['moisture'], 
        bins=[-np.inf, 25, 45, np.inf], 
        labels=["Critical", "Deficit", "Optimal"]
    ).astype(str)

    return df

vellore_farms = load_farm_data()

with st.sidebar:
    st.markdown("<h2 style='color: #fafafa; font-weight: 700; margin-bottom: 0; letter-spacing: -0.05em;'>EcoHydro AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #a1a1aa; font-size: 0.75rem; margin-top: 0; margin-bottom: 24px; font-weight: 600; letter-spacing: 0.05em;'>PRECISION AGRICULTURE OS</p>", unsafe_allow_html=True)
    
    page = st.radio("WORKSPACE", [
        "GIS Farm Telemetry",
        "AI Irrigation Scheduler",
        "Methane & Carbon Ledger",
        "TAM/SAM & Business Model",
    ], label_visibility="collapsed")
    
    st.write("")
    st.write("")
    st.markdown("""
        <div style="background-color: #111113; border: 1px solid #27272a; padding: 16px; border-radius: 8px;">
            <div style="font-size: 0.7rem; color: #a1a1aa; text-transform: uppercase; font-weight: 600;">System Status</div>
            <div style="color: #10b981; font-weight: 500; font-size: 0.9rem; margin-top: 4px; display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%;"></div>
                All Sensors Online
            </div>
            <div style="font-size: 0.7rem; color: #71717a; margin-top: 6px;">Last Sync: Just Now</div>
        </div>
    """, unsafe_allow_html=True)

if page == "GIS Farm Telemetry":
    st.markdown("<div class='main-header'>GIS Farm Telemetry</div>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Monitored", "4,280 Acres", "+12% vs last month")
    col_m2.metric("Freshwater Saved", "18.4M Liters", "38.5% efficiency")
    col_m3.metric("Verified Offsets", "1,420 tCO2e", "AWD Protocol")
    col_m4.metric("Payout Pool", "INR 28.4L", "85% Yield")
    
    st.write("")
    
    if not vellore_farms.empty:
        # Use lambda mapping to prevent Pandas crashing on assigning lists to NaN values
        color_map = {
            "Optimal": [16, 185, 129, 230],     # Emerald
            "Deficit": [245, 158, 11, 230],     # Amber
            "Critical": [239, 68, 68, 230]      # Red
        }
        vellore_farms['color'] = vellore_farms['status'].apply(lambda x: color_map.get(x, [16, 185, 129, 230]))
        
        # Base Scatter Layer (Glow effect)
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="acres * 70",
            pickable=False,
            opacity=0.4
        )
        
        # 3D Column Layer
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=vellore_farms,
            get_position="[lon, lat]",
            get_elevation="acres",
            elevation_scale=100,
            radius=200,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True
        )
        
        view_state = pdk.ViewState(
            latitude=vellore_farms['lat'].mean(),
            longitude=vellore_farms['lon'].mean(),
            zoom=10.2,
            pitch=50,
            bearing=-15
        )
        
        # FULLSCREEN BULLETPROOF TOOLTIP:
        # We put the HTML in the template (allowed) instead of the dataframe (escaped).
        # whiteSpace: nowrap ensures the fullscreen CSS does not squish and crop the text!
        deck_tooltip = {
            "html": "<b style='color: #10b981; font-size: 15px;'>{name}</b><br/><hr style='margin: 8px 0; border-color: #3f3f46;'/><span style='color: #a1a1aa;'>Partner:</span> <b>{fpo}</b><br/><span style='color: #a1a1aa;'>Crop:</span> <b>{crop}</b><br/><span style='color: #a1a1aa;'>Area:</span> <b>{acres} Acres</b><br/><span style='color: #a1a1aa;'>Moisture:</span> <b>{moisture}% ({status})</b>",
            "style": {
                "backgroundColor": "#18181b",
                "color": "#fafafa",
                "border": "1px solid #3f3f46",
                "borderRadius": "8px",
                "padding": "16px",
                "boxShadow": "0 10px 15px -3px rgba(0, 0, 0, 0.6)",
                "zIndex": "2147483647",
                "whiteSpace": "nowrap",
                "lineHeight": "1.6",
                "fontFamily": "'Inter', sans-serif"
            }
        }
        
        deck = pdk.Deck(
            map_provider="carto",
            map_style="dark", 
            layers=[scatter_layer, column_layer],
            initial_view_state=view_state,
            tooltip=deck_tooltip
        )
        
        st.pydeck_chart(deck, width="stretch")
        
        st.write("")
        st.markdown("### Granular HUD Analysis")
        
        farm_names = vellore_farms["name"].unique()
        if len(farm_names) > 0:
            selected_farm_name = st.selectbox("Select Target Farm", farm_names, label_visibility="collapsed")
            farm_subset = vellore_farms[vellore_farms["name"] == selected_farm_name]
            
            if not farm_subset.empty:
                farm = farm_subset.iloc[0]
                temp, hum, precip = get_live_weather(farm['lat'], farm['lon'])
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"""
                    <div class="hud-card status-info">
                        <div>
                            <div class="hud-label">FPO Partner</div>
                            <div class="hud-value">{farm['fpo']}</div>
                        </div>
                        <div>
                            <div class="hud-label">Target Crop</div>
                            <div class="hud-value">{farm['crop']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    status_class = "status-optimal" if farm['moisture'] >= 45 else ("status-deficit" if farm['moisture'] > 25 else "status-critical")
                    st.markdown(f"""
                    <div class="hud-card {status_class}">
                        <div>
                            <div class="hud-label">Soil Moisture</div>
                            <div class="hud-value">{farm['moisture']}% ({farm['status']})</div>
                        </div>
                        <div>
                            <div class="hud-label">Acreage</div>
                            <div class="hud-value">{farm['acres']} Acres</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c3:
                    st.markdown(f"""
                    <div class="hud-card status-optimal">
                        <div>
                            <div class="hud-label">Live Weather (API)</div>
                            <div class="hud-value">{temp} C, {hum}% Hum</div>
                        </div>
                        <div>
                            <div class="hud-label">Protocol Status</div>
                            <div class="hud-value" style="color: #10b981;">Verified Compliant</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

elif page == "AI Irrigation Scheduler":
    st.markdown("<div class='main-header'>AI Irrigation Engine</div>", unsafe_allow_html=True)
    
    if 'api_temp' not in st.session_state: st.session_state.api_temp = 34
    if 'api_hum' not in st.session_state: st.session_state.api_hum = 60
    if 'api_rain' not in st.session_state: st.session_state.api_rain = 5
    
    def fetch_api_data():
        t, h, r = get_live_weather(12.9716, 79.1589)
        st.session_state.api_temp = int(t)
        st.session_state.api_hum = int(h)
        st.session_state.api_rain = int(r)
    
    st.button("Synchronize Live Weather", on_click=fetch_api_data)
    st.write("")
    
    col_inputs, col_results = st.columns([1.2, 1], gap="large")
    with col_inputs:
        with st.expander("Field & Soil Parameters", expanded=True):
            crop_type = st.selectbox("Crop Class", ["Paddy Rice", "Sugarcane", "Groundnut", "Maize"])
            c_acres, c_soil, c_nitro = st.columns(3)
            acres = c_acres.number_input("Area (Acres)", value=10.0)
            soil_m = c_soil.slider("Moisture (%)", 10, 80, 25)
            nitrogen = c_nitro.slider("Nitrogen (mg)", 10, 140, 65)
            
        with st.expander("Meteorological Data", expanded=True):
            c_temp, c_hum, c_rain = st.columns(3)
            temp_in = c_temp.slider("Temp (C)", 20, 45, key='api_temp')
            humidity_in = c_hum.slider("Humidity (%)", 20, 95, key='api_hum')
            rain_in = c_rain.slider("Rain (mm)", 0, 50, key='api_rain')
        
    with col_results:
        input_data = pd.DataFrame([{'N': nitrogen, 'P': 40, 'K': 30, 'Temp': temp_in, 'Humidity': humidity_in, 'pH': 6.8, 'Rainfall': rain_in, 'Soil_Moisture': soil_m}])
        
        if rain_in > 15:
            st.error("Precipitation Override: Irrigation suspended due to expected rainfall.")
            pred_water = 0
        else:
            pred_water = model.predict(input_data)[0] * acres
            
        flood_water = 8500 * acres
        water_saved = max(0, flood_water - pred_water)
        
        st.markdown("### Optimization Output")
        st.metric("Standard Flood Volume", f"{int(flood_water):,} L/Day")
        
        if pred_water == 0:
            st.metric("EcoHydro Drip Requirement", "0 L/Day (Override)", "-100.0%", delta_color="normal")
            progress_pct = 0
        else:
            st.metric("EcoHydro Drip Requirement", f"{int(pred_water):,} L/Day", f"-{round((water_saved/flood_water)*100, 1)}%", delta_color="normal")
            progress_pct = int((pred_water/flood_water)*100)
            
        st.progress(progress_pct)
        
        st.markdown(f"""
        <div style="background: linear-gradient(180deg, #18181b 0%, #111113 100%); border: 1px solid #27272a; padding: 24px; border-radius: 12px; margin-top: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="color: #a1a1aa; font-weight: 600; font-size: 0.75rem; text-transform: uppercase;">Estimated Pumping OPEX Saved</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #10b981; margin-top: 4px;">INR {int(water_saved * 30 * 0.012):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    c_chart, c_explain = st.columns(2)
    with c_chart:
        st.markdown("#### 7-Day Demand Forecast")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig_chart = px.bar(
            pd.DataFrame({'Day': days, 'Flood': [int(flood_water)]*7, 'AI Drip': [int(pred_water * f) for f in [1.0, 0.92, 1.05, 0.88, 0.95, 1.02, 0.90]]}), 
            x='Day', y=['Flood', 'AI Drip'], barmode='group',
            color_discrete_map={"Flood": "#3f3f46", "AI Drip": "#10b981"}
        )
        # Margin fixes to prevent Legend overlapping with X-Axis
        fig_chart.update_layout(
            margin={"r":0,"t":10,"l":0,"b":80}, 
            height=320, 
            legend=dict(yanchor="top", y=-0.3, xanchor="center", x=0.5, orientation="h"), 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="#a1a1aa",
            yaxis=dict(gridcolor="#27272a"),
            xaxis=dict(gridcolor="#27272a")
        )
        st.plotly_chart(fig_chart, width="stretch")
            
    with c_explain:
        st.markdown("#### Feature Importance (XAI)")
        fig_importance = px.bar(
            feature_importance, 
            x='Importance', y='Feature', orientation='h', 
            color='Importance', color_continuous_scale="greens"
        )
        fig_importance.update_layout(
            height=320, 
            margin={"r":0,"t":10,"l":0,"b":40}, 
            showlegend=False, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="#a1a1aa",
            xaxis=dict(gridcolor="#27272a"),
            yaxis=dict(gridcolor="#27272a")
        )
        st.plotly_chart(fig_importance, width="stretch")

elif page == "Methane & Carbon Ledger":
    st.markdown("<div class='main-header'>Carbon Ledger & AWD Audit</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Credits Minted", "1,314 tCO2e", "+200 this week")
    c2.metric("Market Spot Price", "INR 1,530", "+ INR 45")
    c3.metric("Distributed Liquidity", "INR 20.1L", "Completed")
    
    st.write("")
    st.markdown("### Verified Credit Batches")
    ledger_data = pd.DataFrame([
        {"Batch ID": "AWD-26-089", "Entity": "North Arcot Farmers", "Acres": 450, "Volume (tCO2e)": 810, "Value (INR)": 1245000, "Compliance": 98, "Status": "Settled"},
        {"Batch ID": "AWD-26-090", "Entity": "Palar River Basin", "Acres": 280, "Volume (tCO2e)": 504, "Value (INR)": 774000, "Compliance": 100, "Status": "Settled"},
        {"Batch ID": "AWD-26-091", "Entity": "Walajah Organics", "Acres": 150, "Volume (tCO2e)": 270, "Value (INR)": 413100, "Compliance": 92, "Status": "Audit Pending"}
    ])
    
    st.dataframe(
        ledger_data,
        column_config={
            "Batch ID": st.column_config.TextColumn("Batch ID", width="small"),
            "Value (INR)": st.column_config.NumberColumn("Payout Value", format="INR %d"),
            "Compliance": st.column_config.ProgressColumn("Compliance Score", format="%d%%", min_value=0, max_value=100),
            "Status": st.column_config.TextColumn("Status")
        },
        hide_index=True,
        width="stretch"
    )
    
    st.download_button("Export BRSR Compliance Report", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Ledger.csv', mime='text/csv')

elif page == "TAM/SAM & Business Model":
    st.markdown("<div class='main-header'>Market Dynamics & Revenue Projections</div>", unsafe_allow_html=True)
    
    col_tam1, col_tam2, col_tam3 = st.columns(3)
    col_tam1.metric("TAM (Global Smart Ag)", "$12.5B", "Total Addressable")
    col_tam2.metric("SAM (Indian Irrigation)", "$1.8B", "Serviceable Addressable")
    col_tam3.metric("SOM (Year 3 Target)", "$42M", "Serviceable Obtainable")
    
    st.write("---")
    st.markdown("### Revenue Simulator")
    
    calc_c1, calc_c2 = st.columns([1, 1.5], gap="large")
    with calc_c1:
        fpos = st.slider("Active FPO Subscriptions", 5, 300, 30)
        credits = st.slider("Carbon Volume Traded (tCO2e)", 1000, 100000, 15000, step=1000)
        
        saas_rev = fpos * 2500 * 12
        carbon_rev = credits * 1650 * 0.15
        total_arr = saas_rev + carbon_rev
        
    with calc_c2:
        st.markdown(f"""
        <div style="background: linear-gradient(180deg, #18181b 0%, #111113 100%); border: 1px solid #27272a; padding: 32px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4);">
            <div style="color: #a1a1aa; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">Projected Annual Recurring Revenue (ARR)</div>
            <div style="color: #fafafa; font-size: 3.5rem; font-weight: 700; margin: 8px 0 20px 0;">INR {(total_arr)/100000:.2f}L</div>
            <div style="display: flex; gap: 32px; border-top: 1px solid #27272a; padding-top: 20px;">
                <div>
                    <div style="color: #a1a1aa; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px;">SaaS Revenue</div>
                    <div style="color: #fafafa; font-weight: 600; font-size: 1.2rem;">INR {saas_rev/100000:.2f}L</div>
                </div>
                <div>
                    <div style="color: #a1a1aa; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px;">Commission (15%)</div>
                    <div style="color: #10b981; font-weight: 600; font-size: 1.2rem;">INR {carbon_rev/100000:.2f}L</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
