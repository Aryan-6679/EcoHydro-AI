# ... existing code ...
import requests

# --- PAGE CONFIGURATION & PREMIUM UI ---
st.set_page_config(
    page_title="EcoHydro AI - Precision Agriculture Platform",
    page_icon="💧",
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
    
    /* Pitch Box */
    .pitch-box { 
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.02)); 
        border: 1px solid rgba(245, 158, 11, 0.3); 
        padding: 18px; 
        border-radius: 12px; 
        border-left: 5px solid rgba(245, 158, 11, 1);
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 20px;
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
# ... existing code ...
def get_live_weather(lat, lon):
# ... existing code ...
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
with st.sidebar:
    st.image("https://img.icons8.com/color/96/eco-factory.png", width=70)
    st.markdown("<h2 style='margin-top: -10px; margin-bottom: 0px;'>EcoHydro AI</h2>", unsafe_allow_html=True)
    st.caption("Precision Agriculture Engine v2.0")
    
    st.divider()
    page = st.radio("Select Workspace Module", [
        "🌍 GIS Farm Telemetry",
        "⚡ AI Irrigation Scheduler",
        "🍃 Methane & Carbon Ledger",
        "📈 TAM/SAM & Business Model",
    ])
    st.divider()
    
    pitch_toggle = st.checkbox("🏆 Show Judge Pitch Highlights", value=True)
    if pitch_toggle:
        st.markdown("""
        <div class="pitch-box">
            <b>💡 Judge Elevator Pitch (30 Sec):</b><br><br>
            EcoHydro AI eliminates 40% of agricultural freshwater waste using open satellite telemetry and AI, while unlocking up to ₹15,000/acre in corporate carbon credits for smallholder farmers—with <b>Zero Hardware Setup Costs</b>.
        </div>
        """, unsafe_allow_html=True)

# --- PAGE 1: GIS TELEMETRY ---
if page == "🌍 GIS Farm Telemetry":
    st.markdown("<h1 class='gradient-text'>🌍 Vellore Region Precision GIS Telemetry</h1>", unsafe_allow_html=True)
    st.write("Live multispectral soil moisture and crop status across onboarded Farmer Producer Organizations (FPOs).")
    
    st.write("")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Monitored Land", "4,280 Acres", "+12% this month")
    col_m2.metric("Freshwater Saved", "18.4M Liters", "38.5% vs Flood")
    col_m3.metric("Verified CO2 Offsets", "1,420 tCO2e", "AWD Method")
    col_m4.metric("Farmer Payout Pool", "₹28.4 Lakhs", "85% to Farmers")
    
    st.markdown("### 🛰️ Interactive Farm Map View")
    if not vellore_farms.empty:
        # Dynamic compatibility check for newer Plotly versions (6.x/7.x)
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(
                vellore_farms, lat="lat", lon="lon", color="status", size="acres",
                hover_name="name", hover_data=["fpo", "crop", "acres", "moisture"],
                color_discrete_map={"Optimal": "#10b981", "Deficit": "#f59e0b", "Critical": "#ef4444"},
                zoom=10, height=450
            )
            fig_map.update_layout(map_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
        else:
            fig_map = px.scatter_mapbox(
                vellore_farms, lat="lat", lon="lon", color="status", size="acres",
                hover_name="name", hover_data=["fpo", "crop", "acres", "moisture"],
                color_discrete_map={"Optimal": "#10b981", "Deficit": "#f59e0b", "Critical": "#ef4444"},
                zoom=10, height=450
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
            
        try:
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            st.plotly_chart(fig_map) # Fallback for newer Streamlit 1.65.0 deprecations
        
        st.write("")
        st.markdown("### 🔎 Selected Field Telemetry Breakout")
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
                st.success(f"**Live Local Weather:**\n{temp}°C, {hum}% Humidity")
                st.success(f"**AWD Compliance:**\nVerified Compliant")

# --- PAGE 2: AI IRRIGATION SCHEDULER ---
elif page == "⚡ AI Irrigation Scheduler":
    st.markdown("<h1 class='gradient-text'>⚡ XGBoost/RandomForest AI Irrigation Simulator</h1>", unsafe_allow_html=True)
    st.write("Dynamically calculates precise water requirements to eliminate flood irrigation waste.")
    
    if 'api_temp' not in st.session_state: st.session_state.api_temp = 34
    if 'api_hum' not in st.session_state: st.session_state.api_hum = 60
    if 'api_rain' not in st.session_state: st.session_state.api_rain = 5
    
    def fetch_api_data():
        t, h, r = get_live_weather(12.9716, 79.1589) # VIT Vellore Coordinates
        st.session_state.api_temp = int(t)
        st.session_state.api_hum = int(h)
        st.session_state.api_rain = int(r)
        st.toast("Live Vellore Weather Synced!", icon="🌤️")
    
    st.button("🌤️ Auto-Fill Live Vellore Weather (Open-Meteo API)", on_click=fetch_api_data, use_container_width=True)
    st.write("")
    
    col_inputs, col_results = st.columns([1.2, 1])
    with col_inputs:
        with st.expander("🌱 Farm & Soil Parameters", expanded=True):
            crop_type = st.selectbox("Target Crop", ["Paddy Rice", "Sugarcane", "Groundnut", "Maize"])
            c_acres, c_soil, c_nitro = st.columns(3)
            acres = c_acres.number_input("Field Area (Acres)", value=10.0)
            soil_m = c_soil.slider("Soil Moisture (%)", 10, 80, 25)
            nitrogen = c_nitro.slider("Nitrogen (mg/kg)", 10, 140, 65)
            
        with st.expander("🌤️ Live Weather Data", expanded=True):
            c_temp, c_hum, c_rain = st.columns(3)
            temp_in = c_temp.slider("Temp (°C)", 20, 45, key='api_temp')
            humidity_in = c_hum.slider("Humidity (%)", 20, 95, key='api_hum')
            rain_in = c_rain.slider("Rain (mm)", 0, 50, key='api_rain')
        
    with col_results:
        input_data = pd.DataFrame([{'N': nitrogen, 'P': 40, 'K': 30, 'Temp': temp_in, 'Humidity': humidity_in, 'pH': 6.8, 'Rainfall': rain_in, 'Soil_Moisture': soil_m}])
        pred_water = model.predict(input_data)[0] * acres
        flood_water = 8500 * acres
        water_saved = max(0, flood_water - pred_water)
        
        st.markdown("### 💧 Optimization Results")
        st.metric("Traditional Flood Volume", f"{int(flood_water):,} L/Day")
        st.metric("EcoHydro AI Drip Volume", f"{int(pred_water):,} L/Day", f"-{round((water_saved/flood_water)*100, 1)}% Saved", delta_color="normal")
        
        progress_pct = int((pred_water/flood_water)*100)
        st.progress(progress_pct, text=f"AI uses only {progress_pct}% of traditional flood water.")
        st.success(f"**⚡ Monthly Power/Pumping Cost Saved:** ₹{int(water_saved * 30 * 0.012):,}")
    
    st.divider()
    c_chart, c_explain = st.columns(2)
    with c_chart:
        st.markdown("#### 📅 7-Day Water Projection")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig_chart = px.bar(
            pd.DataFrame({'Day': days, 'Flood': [int(flood_water)]*7, 'AI Drip': [int(pred_water * f) for f in [1.0, 0.92, 1.05, 0.88, 0.95, 1.02, 0.90]]}), 
            x='Day', y=['Flood', 'AI Drip'], barmode='group',
            color_discrete_map={"Flood": "#ef4444", "AI Drip": "#10b981"}
        )
        fig_chart.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=300, legend=dict(yanchor="bottom", y=-0.5, xanchor="center", x=0.5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        try: st.plotly_chart(fig_chart, use_container_width=True)
        except Exception: st.plotly_chart(fig_chart)
            
    with c_explain:
        st.markdown("#### 🧠 Explainable AI (XAI) Weights")
        fig_importance = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale="Tealgrn")
        fig_importance.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        try: st.plotly_chart(fig_importance, use_container_width=True)
        except Exception: st.plotly_chart(fig_importance)

# --- PAGE 3: CARBON LEDGER ---
elif page == "🍃 Methane & Carbon Ledger":
    st.markdown("<h1 class='gradient-text'>🍃 Alternate Wetting and Drying (AWD) Carbon Engine</h1>", unsafe_allow_html=True)
    st.write("Verifies methane emission reductions via satellite radar (SAR) to mint high-quality ESG carbon credits.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Credits Minted", "1,314 tCO2e", "+200 this week")
    c2.metric("Market Price per tCO2e", "₹1,530", "+₹45 vs last month")
    c3.metric("Total Farmer Liquidity", "₹20,10,420", "Paid Out")
    
    st.write("")
    st.markdown("### 📋 Verified FPO Carbon Credit Batches")
    ledger_data = pd.DataFrame([
        {"Batch ID": "#AWD-26-089", "FPO Name": "North Arcot Farmers", "Acres": 450, "Credits (tCO2e)": 810, "Value (INR)": 1245000, "Compliance": 98, "Status": "Paid"},
        {"Batch ID": "#AWD-26-090", "FPO Name": "Palar River Basin", "Acres": 280, "Credits (tCO2e)": 504, "Value (INR)": 774000, "Compliance": 100, "Status": "Paid"},
        {"Batch ID": "#AWD-26-091", "FPO Name": "Walajah Organics", "Acres": 150, "Credits (tCO2e)": 270, "Value (INR)": 413100, "Compliance": 92, "Status": "Pending Audit"}
    ])
    
    st.dataframe(
        ledger_data,
        column_config={
            "Batch ID": st.column_config.TextColumn("Batch ID", width="small"),
            "Value (INR)": st.column_config.NumberColumn("Payout Value", format="₹%d"),
            "Compliance": st.column_config.ProgressColumn("AWD Compliance %", format="%d%%", min_value=0, max_value=100),
            "Status": st.column_config.TextColumn("Status")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.download_button("📥 Download SEBI BRSR ESG Report (CSV)", data=ledger_data.to_csv(index=False).encode('utf-8'), file_name='EcoHydro_Carbon_Audit.csv', mime='text/csv', use_container_width=True)

# --- PAGE 4: BUSINESS MODEL ---
elif page == "📈 TAM/SAM & Business Model":
    st.markdown("<h1 class='gradient-text'>📈 Financial Projections & Market Sizing</h1>", unsafe_allow_html=True)
    st.write("EcoHydro AI leverages a dual-revenue stream: B2B SaaS for FPOs and a 15% marketplace commission on carbon credits.")
    
    st.write("")
    col_tam1, col_tam2, col_tam3 = st.columns(3)
    col_tam1.metric("TAM (Smart Ag & Offsets)", "$12.5 Billion", "Global")
    col_tam2.metric("SAM (Indian Irrigation Automation)", "$1.8 Billion", "South India Focus")
    col_tam3.metric("SOM (Year 3 Projection)", "$42 Million", "Serviceable Obtainable")
    
    st.write("---")
    st.markdown("### 🧮 Interactive Monetization Calculator")
    
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
            <h1 style="color: #10b981; font-size: 3rem; margin: 10px 0;">₹{(total_arr)/100000:.2f} Lakhs/Yr</h1>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 1.1rem;">
                <span><b>SaaS Subscriptions:</b> ₹{saas_rev/100000:.2f}L</span>
                <span><b>Carbon Commission:</b> ₹{carbon_rev/100000:.2f}L</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
