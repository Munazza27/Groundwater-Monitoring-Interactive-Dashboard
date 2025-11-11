An interactive, map-based dashboard built with Streamlit and Folium to visualize groundwater trends across Indian states. It overlays Google-style maps with state-level groundwater data, helping identify regions with critical water level decline. The dashboard supports decision-making

--->Youtube link-https://youtu.be/RWnC0_IzAc4 <----

-->Features:
   State-wise drill-down metrics for groundwater rise, fall, and stability
   Choropleth overlay showing % of wells in fall category
   Forecasting prototype for future groundwater trends
   Alerts for states with critical groundwater depletion
   National summary pie chart and Top 5 fall states

--->Used:
Python           
Streamlit        
Folium           
streamlit-folium 
Pandas           
Plotly           
Requests                  


Map Integration
Uses Google Maps tiles via Folium’s custom TileLayer:


and since its written in stram lit to run it
streamlit run groundwater_dashboard.py
