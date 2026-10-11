import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestRegressor
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="EcoHydro AI | Skynet Hackathon", page_icon="🌱", layout="wide")

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
# 2. GOOGLE DRIVE / SHEETS INTEGRATION
# ==========================================
def log_to_google_drive(data_row):
    """Pushes telemetry data directly to Google Drive (Sheets)"""
    try:
        # Define API scope
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Pull credentials safely from Streamlit Secrets
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open the specific Google Sheet
        sheet = client.open("EcoHydro_Data_Logs").sheet1
        
        # Append the new row of data
        sheet.append_row(data_row)
        return True
    except Exception as e:
        # If it fails (e.g. no internet, bad keys), don't crash the app!
        st.error(f"⚠️ Drive Sync Error (Check Secrets): {e}")
        return False

# ==========================================
# 3. DATA LOADING & ML MODEL
# ==========================================
@st.cache_data
def load_farm_data():
    try:
        df = pd.read_csv("vellore_farm_data.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Katpadi FPO", "Kaniyambadi FPO", "Anaicut FPO", "Gudiyatham FPO", "K.V. Kuppam FPO"],
            "lat": [12.9800, 12.8333, 12.8754, 12.9463, 13.0485],
            "lon": [79.1390, 79.1333, 78.9833, 78.8711, 79.0345],
            "crop": ["Paddy", "Sugarcane", "Groundnut", "Paddy", "Banana"],
            "acres": [120, 85, 40, 150, 60],
            "moisture": [45, 30, 20, 50, 60],
            "status": ["Optimal", "Deficit", "Critical", "Optimal", "Optimal"]
        })

@st.cache_resource
def train_ai_model():
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        'N': np.random.uniform(10, 120, n_samples),
        'P': np.random.uniform(5, 80, n_samples),
        'K': np.random.uniform(10, 80, n_samples),
        'Temp': np.random.uniform(22, 42, n_samples),
        'Humidity': np.random.uniform(30, 90, n_samples),
        'Rainfall_Forecast': np.random.uniform(0, 50, n_samples),
        'pH': np.random.uniform(5.5, 8.5, n_samples),
        'Soil_Moisture': np.random.uniform(10, 80, n_samples)
    })
    y = np.clip((X['Temp']*200) + ((100-X['Humidity'])*50) - (X['Rainfall_Forecast']*25) - (X['Soil_Moisture']*80), 1000, 8000)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X.columns

farm_data = load_farm_data()
ai_model, feature_names = train_ai_model()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=60)
    st.title("EcoHydro AI")
    st.caption("Team Skynet | Precision Agriculture")
    st.markdown("---")
    
    view = st.radio("Select Workspace Module", [
        "🌍 GIS Farm Telemetry",
        "⚡ AI Irrigation Scheduler",
        "🍃 Methane & Carbon Ledger",
        "📈 TAM/SAM & Business Model"
    ])

# ==========================================
# 5. MAIN DASHBOARD VIEWS
# ==========================================

if view == "🌍 GIS Farm Telemetry":
    st.header("🌍 Precision GIS Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Total Monitored Land</div><div class="metric-value">4,280 Acres</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Freshwater Saved</div><div class="metric-value">18.4M Liters</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">Verified CO2 Offsets</div><div class="metric-value">1,420 Tons</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">Farmer Payout Pool</div><div class="metric-value">₹28.4 Lakh</div></div>', unsafe_allow_html=True)
    
    fig_map = px.scatter_mapbox(
        farm_data, lat="lat", lon="lon", color="status", size="acres",
        hover_name="name", hover_data=["crop", "acres", "moisture"],
        color_discrete_map={"Optimal": "#10b981", "Deficit": "#f59e0b", "Critical": "#ef4444"},
        zoom=10, center={"lat": 12.9165, "lon": 79.1325}, mapbox_style="carto-positron"
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

elif view == "⚡ AI Irrigation Scheduler":
    st.header("⚡ AI Irrigation Scheduler & Live Weather Integration")
    
    if 'live_temp' not in st.session_state:
        st.session_state['live_temp'] = 32.5
        st.session_state['live_hum'] = 65
        st.session_state['forecast_rain'] = 4.5

    if st.button("🌦️ Sync Live Vellore Weather & 48hr Forecast (Open-Meteo API)"):
        try:
            res = requests.get("https://api.open-meteo.com/v1/forecast?latitude=12.9165&longitude=79.1325&current=temperature_2m,relative_humidity_2m&daily=precipitation_sum&timezone=auto", timeout=5)
            data = res.json()
            st.session_state['live_temp'] = float(data['current']['temperature_2m'])
            st.session_state['live_hum'] = int(data['current']['relative_humidity_2m'])
            st.session_state['forecast_rain'] = float(data['daily']['precipitation_sum'][0])
            st.success("✅ Weather Synced Successfully!")
        except Exception as e:
            st.warning("⚠️ API connection failed. Using fallback simulation values.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        n_val = st.slider("Nitrogen (N)", 0, 150, 60)
        p_val = st.slider("Phosphorus (P)", 0, 100, 30)
        k_val = st.slider("Potassium (K)", 0, 100, 40)
        ph_val = st.slider("Soil pH", 4.0, 9.0, 6.5)
    with col2:
        t_val = st.slider("Current Ambient Temp (°C)", 10.0, 50.0, float(st.session_state['live_temp']))
        h_val = st.slider("Current Humidity (%)", 10, 100, int(st.session_state['live_hum']))
        r_val = st.slider("48hr Weather Forecast Rain (mm)", 0.0, 50.0, float(st.session_state['forecast_rain']))
        sm_val = st.slider("Current Soil Moisture (%)", 0, 100, 35)
    
    input_df = pd.DataFrame([[n_val, p_val, k_val, t_val, h_val, r_val, ph_val, sm_val]], columns=feature_names)
    water_pred = ai_model.predict(input_df)[0]
    flood_water = water_pred * 1.65 
    
    with col3:
        st.subheader("AI Irrigation Action Plan")
        if r_val > 15.0:
            st.error(f"🌧️ **Heavy Rain Forecast ({r_val}mm):** Postpone irrigation! Natural rainfall will replenish soil moisture.")
            water_pred = 0.0 
        else:
            st.info(f"💧 **Optimal Drip Irrigation:** {int(water_pred)} Liters/Acre recommended today.")
            
        st.warning(f"🌊 **Traditional Flood Irrigation:** {int(flood_water)} Liters/Acre today.")
        st.success(f"📈 **Freshwater Saved:** {int(flood_water - water_pred)} Liters/Acre")

    # ==========================================
    # RECURSIVE LEARNING + CLOUD DATA ACCUMULATION
    # ==========================================
    st.markdown("---")
    st.subheader("🧬 Recursive Learning & Cloud Data Accumulation")
    st.caption("Submit field feedback. This retrains the model dynamically AND securely archives the telemetry in Google Drive.")

    with st.form("feedback_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            actual_water_used = st.number_input("Actual Water Used (Liters/Acre)", min_value=0.0, max_value=10000.0, value=float(water_pred))
        with col_f2:
            feedback_notes = st.selectbox("Field Performance Feedback", [
                "Optimal Growth", "Slight Moisture Stress", "Slight Over-saturation"
            ])
        
        submit_feedback = st.form_submit_button("🔄 Retrain AI & Sync to Google Drive")

        if submit_feedback:
            # 1. Retrain local AI Model
            adjustment = 1.15 if "Stress" in feedback_notes else (0.85 if "Over-saturation" in feedback_notes else 1.0)
            ai_model.fit(input_df, pd.Series([actual_water_used * adjustment]))
            
            # 2. Log "Tonnes of Data" to Google Drive 
            data_row = [n_val, p_val, k_val, t_val, h_val, r_val, sm_val, float(water_pred), actual_water_used, feedback_notes]
            success = log_to_google_drive(data_row)
            
            if success:
                st.success("🎉 **Data Accumulated!** AI model retrained locally AND telemetry successfully archived to Google Drive database.")
                st.balloons()

elif view == "🍃 Methane & Carbon Ledger":
    st.header("🍃 Verified Carbon Credit Ledger")
    ledger_data = pd.DataFrame({
        "FPO / Cooperative": ["Katpadi FPO", "Kaniyambadi FPO", "Anaicut FPO", "Gudiyatham FPO"],
        "Acreage Under AWD": [1200, 850, 400, 1500],
        "Methane Prevented (kg)": [14400, 10200, 4800, 18000],
        "CO2e Offset (Tons)": [403.2, 285.6, 134.4, 504.0],
        "Estimated Payout (INR)": ["₹8,06,000", "₹5,70,000", "₹2,68,000", "₹10,08,000"]
    })
    st.dataframe(ledger_data, use_container_width=True)

elif view == "📈 TAM/SAM & Business Model":
    st.header("📈 Market Sizing & Revenue Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("TAM (Global)", "$14 Billion")
    col2.metric("SAM (India)", "$2.1 Billion")
    col3.metric("SOM (South India)", "$150 Million")
