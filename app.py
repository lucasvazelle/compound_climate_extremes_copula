import os
import xarray as xr
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# Charger un fichier NetCDF
def load_netcdf_data(nc_file):
    try:
        return xr.open_dataset(nc_file, engine="netcdf4")
    except Exception as e:
        print(f"Erreur lors du chargement de {nc_file}: {e}")
        return None

# Charger tous les fichiers NetCDF d'un répertoire et sous-répertoires
def load_all_netcdf_files(directory):
    datasets, filenames = [], []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith('.nc'):
                path = os.path.join(root, fname)
                ds = load_netcdf_data(path)
                if ds is not None:
                    datasets.append(ds)
                    filenames.append(path)
    return datasets, filenames

# Extraction d'un titre lisible

def extract_title(filename):
    base = os.path.basename(filename)
    # Extrait le segment après le premier underscore jusqu'au premier tiret
    start = base.find('_') + 1
    end = base.find('-', start)
    title = base[start:end]
    # Si un segment entre "monthly-" et "-grid" existe, on l'ajoute après un underscore
    m_prefix = 'monthly-'
    g_suffix = '-grid'
    m_idx = base.find(m_prefix)
    if m_idx != -1:
        m_start = m_idx + len(m_prefix)
        g_idx = base.find(g_suffix, m_start)
        if g_idx != -1:
            extra = base[m_start:g_idx]
            title = f"{title}_{extra}"
    return title

# Sélection du thème de couleur selon l'indicateur
# Couleurs moins vives pour dry et wind
def select_theme(filename):
    fn = filename.lower()
    if 'frost' in fn or 'precipitation' in fn:
        return 'Blues'
    elif 'hot' in fn or 'temperature' in fn or 'drought' in fn:
        return 'Reds'
    elif 'dry' in fn or 'wind' in fn:
        return 'Greens'  # moins vives, ton plus doux
    else:
        return 'Viridis'

# Chargement initial des datasets
datasets, filenames = load_all_netcdf_files('./')
titles = [extract_title(f) for f in filenames]

# Création de l'app Dash
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Cartes des indicateurs climatiques"),
    html.Label("Année:"),
    dcc.Slider(
        id='year-slider', min=1979, max=2023, step=1, value=2023,
        marks={y: str(y) for y in range(1979, 2024, 5)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Tabs(
        id='tabs', value='0',
        children=[dcc.Tab(label=t, value=str(i)) for i, t in enumerate(titles)]
    ),
    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'), Input('year-slider', 'value')]
)
def render_tab(tab_idx, year):
    # Sélection du dataset actif
    i = int(tab_idx)
    ds = datasets[i]
    filename = filenames[i]

    # Nom de la première variable
    var = list(ds.data_vars)[0]

    # Extraction des données pour l'année
    data = ds[var].sel(time=str(year)).load()
    df = data.to_dataframe().reset_index().dropna().rename(columns={var: 'valeur'})

    # Bornes pour l'échelle de couleur
    vmin, vmax = df['valeur'].min(), df['valeur'].max()

    # Création de la figure avec points colorés par valeur
    fig = go.Figure(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        customdata=df['valeur'],  # pour hover
        marker=dict(
            size=6,
            color=df['valeur'],
            colorscale=select_theme(filename),
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title="Valeur", thickness=15, ticks="outside"),
            opacity=0.1  # faible opacité pour voir la carte
        ),
        hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Valeur: %{customdata:.2f}<extra></extra>"
    ))

    # Centrer la carte sur l'étendue des données
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return html.Div([
        html.H2(f"{extract_title(filename)} - {year}"),
        dcc.Graph(figure=fig, style={'height':'80vh'})
    ])

if __name__ == '__main__':
    app.run(debug=True)
