import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import json, os, requests
import plotly.express as px
import pydeck as pdk
from sklearn.linear_model import LinearRegression
from io import StringIO

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(page_title="Groundwater Monitoring Dashboard", layout="wide")
st.title("Groundwater Monitoring Dashboard - India")

# ----------------------
# LOAD GEOJSON
# ----------------------
geojson_path = "india_states.geojson"
if not os.path.exists(geojson_path):
    url = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_STATES.geojson"
    r = requests.get(url)
    if r.status_code == 200:
        with open(geojson_path, "wb") as f:
            f.write(r.content)
    else:
        st.error(f"Could not download GeoJSON file. HTTP status: {r.status_code}")
        st.stop()

with open(geojson_path, "r", encoding="utf-8") as f:
    india_states_geojson = json.load(f)

first_props = india_states_geojson["features"][0]["properties"]
state_key = next((k for k in first_props.keys() if "name" in k.lower()), None)
if not state_key:
    st.error("Could not find a state name key in GeoJSON properties.")
    st.stop()

# ----------------------
# SAMPLE DATA OR UPLOAD
# ----------------------
sample_data = """State,Total_Wells,Rise_0_2,Rise_2_4,Rise_4+,Fall_0_2,Fall_2_4,Fall_4+,NoChange_0_2,NoChange_2_4,NoChange_4+,Lat,Lon
Maharashtra,100,20,10,5,30,15,10,5,3,2,19.7515,75.7139
Gujarat,80,15,8,4,25,12,6,5,3,2,22.2587,71.1924
Tamil Nadu,90,18,9,6,28,14,8,4,2,1,11.1271,78.6569
Karnataka,85,16,7,5,26,13,7,6,3,2,15.3173,75.7139
Rajasthan,95,19,10,6,29,15,9,3,2,2,27.0238,74.2179
"""

uploaded_file = st.file_uploader("Upload Groundwater CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(StringIO(sample_data))

df['State'] = df['State'].str.strip().str.title()

# ----------------------
# CALCULATIONS
# ----------------------
df['Rise_Total'] = df[['Rise_0_2','Rise_2_4','Rise_4+']].sum(axis=1)
df['Fall_Total'] = df[['Fall_0_2','Fall_2_4','Fall_4+']].sum(axis=1)
df['NoChange_Total'] = df[['NoChange_0_2','NoChange_2_4','NoChange_4+']].sum(axis=1)
df['Fall_Percent'] = (df['Fall_Total']/df['Total_Wells'])*100

# Forecast models (simple linear over index)
X = np.arange(len(df)).reshape(-1,1)
rise_model = LinearRegression().fit(X, df['Rise_Total'])
fall_model = LinearRegression().fit(X, df['Fall_Total'])
df['Forecast_Rise'] = rise_model.predict(X).astype(int)
df['Forecast_Fall'] = fall_model.predict(X).astype(int)

# National totals
total_wells = df['Total_Wells'].sum()
total_rise = df['Rise_Total'].sum()
total_fall = df['Fall_Total'].sum()
total_no_change = df['NoChange_Total'].sum()

# ----------------------
# KPI ROW
# ----------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Wells Nationwide", total_wells)
col2.metric("Total Rise Wells", total_rise)
col3.metric("Total Fall Wells", total_fall)

# ----------------------
# MAPS (Tabs) - Pydeck first
# ----------------------
tab_pydeck, tab_choro = st.tabs(["3D Pydeck Map", "Google Map Choropleth"])

with tab_pydeck:
    # Build point cloud for HexagonLayer
    well_points = []
    for _, row in df.iterrows():
        lat, lon = row['Lat'], row['Lon']
        for _ in range(int(row['Rise_Total'])):
            well_points.append({'Lat': lat + np.random.uniform(-0.05,0.05),
                                'Lon': lon + np.random.uniform(-0.05,0.05),
                                'Category': 'Rise'})
        for _ in range(int(row['Fall_Total'])):
            well_points.append({'Lat': lat + np.random.uniform(-0.05,0.05),
                                'Lon': lon + np.random.uniform(-0.05,0.05),
                                'Category': 'Fall'})
        for _ in range(int(row['NoChange_Total'])):
            well_points.append({'Lat': lat + np.random.uniform(-0.05,0.05),
                                'Lon': lon + np.random.uniform(-0.05,0.05),
                                'Category': 'No Change'})
    well_df = pd.DataFrame(well_points)

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=4.5, pitch=50),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=well_df,
                get_position='[Lon, Lat]',
                auto_highlight=True,
                radius=2000,
                elevation_scale=500,
                elevation_range=[0, 10000],
                pickable=True,
                extruded=True,
            )
        ],
        tooltip={"text":"# Wells: {elevationValue}"}
    )
    st.pydeck_chart(deck, use_container_width=True)

with tab_choro:
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles=None)
    google_tiles = folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google', name='Google Maps', overlay=False, control=True
    )
    google_tiles.add_to(m)

    folium.Choropleth(
        geo_data=india_states_geojson,
        name="Groundwater Fall",
        data=df,
        columns=["State", "Fall_Percent"],
        key_on=f"feature.properties.{state_key}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="% Wells in Fall"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1600, height=800)

# ----------------------
# STATE DRILL-DOWN
# ----------------------
st.subheader("State Drill-Down")
state_list = df['State'].tolist()
selected_state = st.selectbox("Select a State", state_list)
s = df[df['State']==selected_state].iloc[0]

st.metric("Total Wells", int(s['Total_Wells']))
st.metric("% Rise", f"{s['Rise_Total']/s['Total_Wells']*100:.2f}%")
st.metric("% Fall", f"{s['Fall_Total']/s['Total_Wells']*100:.2f}%")
st.metric("% No Change", f"{s['NoChange_Total']/s['Total_Wells']*100:.2f}%")
st.metric("Forecast Rise Wells", int(s['Forecast_Rise']))
st.metric("Forecast Fall Wells", int(s['Forecast_Fall']))

# ----------------------
# NATIONAL SUMMARY
# ----------------------
st.subheader("National Summary")
fig_pie = px.pie(
    names=['Rise', 'Fall', 'No Change'],
    values=[total_rise, total_fall, total_no_change],
    title="Proportion of Wells Nationwide"
)
st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------
# TOP 5 STATES
# ----------------------
st.subheader("Top 5 States - Highest Groundwater Fall (>4m)")
top5_fall = df[['State', 'Fall_4+']].sort_values('Fall_4+', ascending=False).head(5)
st.table(top5_fall)

# ----------------------
# CRITICAL ALERTS
# ----------------------
st.subheader("Critical Alerts")
critical_states = df[df['Fall_Percent'] > 50]['State'].tolist()
if critical_states:
    st.warning(f"States with >50% wells in fall category: {', '.join(critical_states)}")
else:
    st.success("No critical states currently")

# ----------------------
# POLICY SANDBOX SIMULATION
# ----------------------
st.subheader("Policy Sandbox: Simulate Interventions")
st.write(
    "Experiment with different interventions to see how they might affect groundwater outcomes. "
    "Adjust rainfall and conservation adoption, and the dashboard will simulate new Rise/Fall values."
)

col_rain, col_conserve = st.columns(2)
rainfall_change = col_rain.slider("Rainfall Change (%)", -50, 50, 0, step=5)
conservation_adoption = col_conserve.slider("Conservation Adoption (%)", 0, 100, 0, step=10)

df_sim = df.copy()
df_sim['Sim_Rise'] = df_sim['Rise_Total'] * (1 + rainfall_change/100)
df_sim['Sim_Fall'] = df_sim['Fall_Total'] * (1 - conservation_adoption/100)
df_sim['Sim_NoChange'] = df_sim['NoChange_Total']

sim_rise_total = df_sim['Sim_Rise'].sum()
sim_fall_total = df_sim['Sim_Fall'].sum()
sim_nochange_total = df_sim['Sim_NoChange'].sum()

col_sr, col_sf, col_nc = st.columns(3)
col_sr.metric("Simulated Rise Wells", f"{int(sim_rise_total)}", delta=f"{rainfall_change}% rainfall")
col_sf.metric("Simulated Fall Wells", f"{int(sim_fall_total)}", delta=f"-{conservation_adoption}% fall")
col_nc.metric("No Change Wells", int(sim_nochange_total))

st.subheader("Simulation Results by State")
sim_chart = px.bar(
    df_sim,
    x="State",
    y=["Sim_Rise", "Sim_Fall", "Sim_NoChange"],
    barmode="group",
    title="Simulated Well Outcomes by State"
)
st.plotly_chart(sim_chart, use_container_width=True)

# ----------------------
# INTERACTIVE GROUNDWATER EXPLORATION (counts)
# ----------------------
st.subheader("Interactive Groundwater Data Exploration")
st.write(
    "Use the controls below to explore groundwater well totals by state. "
    "Select multiple states and optionally smooth the data with a rolling average."
)

all_states = df['State'].tolist()
with st.container(border=True):
    selected_states = st.multiselect("Select States", all_states, default=all_states)
    rolling_average_counts = st.toggle("Apply Rolling Average (3-period) to totals", value=False)

plot_counts = df.set_index("State")[["Rise_Total", "Fall_Total", "NoChange_Total"]].loc[selected_states].T
if rolling_average_counts:
    plot_counts = plot_counts.rolling(3, axis=1).mean().dropna(axis=1)

tab_counts_chart, tab_counts_df = st.tabs(["Chart (Totals)", "Dataframe (Totals)"])
with tab_counts_chart:
    st.line_chart(plot_counts, height=300)
with tab_counts_df:
    st.dataframe(plot_counts, use_container_width=True, height=300)

# ----------------------
# INTERACTIVE GROUNDWATER EXPLORATION (percentages)
# ----------------------
st.subheader("Interactive Groundwater Percentage Comparison")
st.write("Compare states by relative percentages. This normalizes for different total well counts.")

df_pct = df.copy()
df_pct['Rise_Pct'] = df_pct['Rise_Total'] / df_pct['Total_Wells'] * 100
df_pct['Fall_Pct'] = df_pct['Fall_Total'] / df_pct['Total_Wells'] * 100
df_pct['NoChange_Pct'] = df_pct['NoChange_Total'] / df_pct['Total_Wells'] * 100

with st.container(border=True):
    selected_states_pct = st.multiselect("Select States (Percentages)", all_states, default=all_states, key="pct_states")
    rolling_average_pct = st.toggle("Apply Rolling Average (3-period) to percentages", value=False, key="pct_toggle")

plot_pct = df_pct.set_index("State")[["Rise_Pct", "Fall_Pct", "NoChange_Pct"]].loc[selected_states_pct].T
if rolling_average_pct:
    plot_pct = plot_pct.rolling(3, axis=1).mean().dropna(axis=1)

tab_pct_chart, tab_pct_df = st.tabs(["Chart (Percentages)", "Dataframe (Percentages)"])
with tab_pct_chart:
    st.line_chart(plot_pct, height=300)
with tab_pct_df:
    st.dataframe(plot_pct.round(2), use_container_width=True, height=300)