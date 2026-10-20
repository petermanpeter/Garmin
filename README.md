# Garmin program description
analyze Garmin activities and weight

# Update v20261020b
- Basemap default is now truly OpenStreetMap (the CartoDB layer is only
  listed in the layer switcher, not shown first).
- Mouse-wheel zoom is off on the maps, so the page can be scrolled back
  up to the tabs while the pointer is over a map (zoom with +/− buttons).
- The **Map** tab is renamed **Specific Run**.
- New **All Run** tab: every run in the chosen period layered on one
  map in semi-transparent blue — overlapping routes stack darker, so
  darkest blue = most frequently run route. Routes are fetched once
  (button) and cached in the session.

# Map: OpenStreetMap (since v20261020)
The route map in the **Map** tab is drawn with Leaflet (folium /
streamlit-folium) directly on OpenStreetMap tiles
(`https://tile.openstreetmap.org/{z}/{x}/{y}.png`, © OpenStreetMap
contributors). **No Google Maps or Mapbox API key/token is needed.**

Why it changed: the map previously used Plotly's Mapbox API
(`px.line_mapbox` / `add_scattermapbox` / `mapbox_style="carto-positron"`),
which was removed in Plotly 6+, so the map could not render on a fresh
install. The OSM map keeps the same features: full route in red, selected
segment in green, blue start / red end markers, numbered km markers, and
the segment distance / slope metrics + elevation profile below it. A layer
switcher (top-right of the map) also offers a CartoDB light basemap, which
renders the same OpenStreetMap data.

# Install Command
pip install dash plotly pandas openpyxl streamlit
pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
#pip install selenium webdriver-manager requests gpxpy folium pandas python-dateutil tqdm

python App.py
streamlit run App.py
python RaceResult.py
streamlit run RaceResult.py

# Create requirements.txt to hold the library
pip freeze > requirements.txt  

# Commit
git add App.py Weight_20260111.xlsx requirements.txt README.md
git commit -m "v20260403b: add route selection to view distance/time/slope map"  
git push  

