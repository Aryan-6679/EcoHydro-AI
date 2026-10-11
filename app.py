# ... existing code ...
@st.cache_data
def load_farm_data():
    """Load from CSV, or generate realistic fallback data if CSV is missing."""
    try:
        df = pd.read_csv("vellore_farm_data.csv")
        if not df.empty:
            # Ensure required columns exist for the tooltip to prevent rendering errors
            if 'fpo' not in df.columns: df['fpo'] = "Vellore Agri Hub"
            if 'crop' not in df.columns: df['crop'] = "Mixed Crop"
            if 'name' not in df.columns: df['name'] = [f"Farm-0{i}" for i in range(len(df))]
            if 'moisture' not in df.columns: df['moisture'] = 45.0
            if 'acres' not in df.columns: df['acres'] = 10.0
            if 'status' not in df.columns:
                df['status'] = pd.cut(df['moisture'], bins=[0, 25, 45, 100], labels=["Critical", "Deficit", "Optimal"]).astype(str)
            return df
    except FileNotFoundError:
        pass
    
    # Fallback to ensure the map always looks incredible even without the CSV
# ... existing code ...
```

```python:EcoHydro AI Dashboard:app.py
# ... existing code ...
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
                "html": "<div style='width: 220px; white-space: normal; line-height: 1.5; word-wrap: break-word;'><b>{name}</b><hr style='margin: 6px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.15);' /><b>FPO:</b> {fpo}<br/><b>Crop:</b> {crop}<br/><b>Area:</b> {acres} Acres<br/><b>Moisture:</b> {moisture}% ({status})</div>",
                "style": {
                    "backgroundColor": "rgba(15, 23, 42, 0.95)",
                    "color": "#f8fafc",
                    "borderRadius": "8px",
                    "padding": "12px",
                    "border": "1px solid rgba(52, 211, 153, 0.4)",
                    "boxShadow": "0 10px 25px rgba(0,0,0,0.5)",
                    "fontFamily": "sans-serif",
                    "fontSize": "13px"
                }
            }
        )
        
        # Render PyDeck
# ... existing code ...
