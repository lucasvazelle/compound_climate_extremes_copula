import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import xarray as xr
from import_data import ImportData

# --- Charger et préparer ds_all ---
importer = ImportData()
ds_rh   = importer.import_relative_humidity()
ds_temp = importer.import_temperature()
ds_u    = importer.import_u_wind()
ds_v    = importer.import_v_wind()

# Sélection au niveau 1000 hPa et renommage
dry_da = ds_rh["r"].rename("dry")
hot_da = ds_temp["t"].rename("hot")
hot_da = hot_da - 273.15  # Kelvin -> °C

u_da = ds_u["u"]
v_da = ds_v["v"]
wind_da = np.sqrt(u_da**2 + v_da**2).rename("wind")  # module du vent

# Fusion en un seul Dataset
ds_all = xr.merge([dry_da, hot_da, wind_da])

# Calcul de l'indicateur HDW (Heat-Dry-Wind)
hdw = xr.where(
    (ds_all['hot'] > 35) &
    (ds_all['dry'] < 30) &
    (ds_all['wind'] >= 7),
    1, 0
).rename('HDW')

# Titre de l'application
st.title("Carte journalière des indicateurs et HDW")

# Sélection de la date
min_date = pd.to_datetime(str(ds_all['valid_time'].min().values))
max_date = pd.to_datetime(str(ds_all['valid_time'].max().values))
selected_date = st.date_input(
    "Choisissez une date", 
    value=min_date.date(), 
    min_value=min_date.date(), 
    max_value=max_date.date()
)
sel = np.datetime64(selected_date)

# Choix de l'indicateur à afficher
var_sel = st.selectbox("Choisir un indicateur", ['dry', 'hot', 'wind'], index=1)

# Extraire les données pour la date sélectionnée
selected_da = ds_all.sel(valid_time=sel)

df = selected_da[var_sel].to_dataframe().reset_index().dropna(subset=[var_sel])

# Définir une palette simple
palettes = {
    'dry': [[144, 238, 144, 50], [60, 179, 113, 100], [32, 178, 170, 150], [0, 128, 128, 200], [0, 100, 0, 220], [0, 128, 255, 255]],
    'hot': [[255, 182, 193, 50], [255, 99, 71, 100], [255, 69, 0, 150], [255, 0, 0, 200], [178, 34, 34, 220], [139, 0, 0, 255]],
    'wind': [[173, 216, 230, 50], [135, 206, 235, 100], [0, 191, 255, 150], [30, 144, 255, 200], [0, 0, 255, 220], [0, 0, 139, 255]]
}

# Heatmap layer for selected indicator
layer_heat = pdk.Layer(
    'HeatmapLayer',
    data=df,
    get_position=['longitude', 'latitude'],  # correct order lon, lat
    get_weight=var_sel,
    radiusPixels=30,
    opacity=0.6,
    colorRange=palettes[var_sel]
)

# Scatter layer for selected indicator (circles sized by value)
# Compute a radius column normalized to highlight values
df['radius'] = ((df[var_sel] - df[var_sel].min()) /
                 (df[var_sel].max() - df[var_sel].min() + 1e-6)) * 20000 + 1000
indicator_layer = pdk.Layer(
    'ScatterplotLayer',
    data=df,
    get_position=['longitude', 'latitude'],
    get_radius='radius',
    get_fill_color=[0, 0, 255, 140],  # semi-transparent blue
    pickable=True
)

# Points HDW en rouge
hdw_day = hdw.sel(valid_time=sel)
df_hdw = hdw_day.where(hdw_day == 1).to_dataframe().reset_index().dropna(subset=['HDW'])
layer_hdw = pdk.Layer(
    'ScatterplotLayer',
    data=df_hdw,
    get_position=['longitude', 'latitude'],
    get_color=[255, 0, 0, 200],
    get_radius=15000,
    pickable=False
)

# Compose layers: indicator layer first, then heatmap, then HDW points
hdw_day = hdw.sel(valid_time=sel)
df_hdw = hdw_day.where(hdw_day == 1).to_dataframe().reset_index().dropna(subset=['HDW'])
layer_hdw = pdk.Layer(
    'ScatterplotLayer',
    data=df_hdw,
    get_position=['longitude', 'latitudet'],
    get_color=[255, 0, 0, 200],
    get_radius=15000,
    pickable=False
)

# Vue initiale centrée sur la zone moyenne
avg_lat = float(df['latitude'].mean())
avg_lon = float(df['longitude'].mean())
view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=4, pitch=35)

# Affichage pydeck
r = pdk.Deck(
    layers=[layer_heat, layer_hdw],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v9'
)
st.pydeck_chart(r)

# Affichage des événements HDW détectés
if not df_hdw.empty:
    st.subheader(f"Événements HDW le {selected_date}")
    st.dataframe(df_hdw[['valid_time', 'latitude', 'longitude']].rename(columns={'valid_time': 'date'}))
else:
    st.write(f"Aucun épisode HDW détecté le {selected_date}.")
