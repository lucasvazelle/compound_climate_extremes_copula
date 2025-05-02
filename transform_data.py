import numpy as np
import xarray as xr
from import_data import ImportData
 
def transformation_unite_de_mesure_data(importer: ImportData) -> xr.Dataset:
    """
    Charge les données RH, T, U, V via ImportData :
      - dry  : humidité relative [%]
      - hot  : température en °C
      - wind : vitesse du vent en m/s
    """
    # Charger les séries multi-années
    ds_rh   = importer.import_relative_humidity()
    ds_temp = importer.import_temperature()
    ds_u    = importer.import_u_wind()
    ds_v    = importer.import_v_wind()

    # Sélection au niveau=1000 hPa
    dry = ds_rh["r"].rename("dry")
    t_da = ds_temp["t"]
    u_da = ds_u["u"]
    v_da = ds_v["v"]

    # 3) Transformations :
    #    - température : K → °C
    #    - vent     : composantes U/V → module
    hot = (t_da - 273.15).rename("hot")
    wind = np.sqrt(u_da**2 + v_da**2).rename("wind")

    return dry, hot, wind
