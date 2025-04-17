import os
import time
import pandas as pd
import netCDF4 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
class ImportData:
    def __init__(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _get_full_path(self, relative_path: str) -> str:
        """
        Construit le chemin absolu en joignant le répertoire de base et le chemin relatif.
        """
        return os.path.join(self.base_path, relative_path)

    def import_data_csv(self, file_path, sep=',', chunksize=1000, encoding='utf-8'):
        full_path = self._get_full_path(file_path)
        s_time_chunk = time.time()
        chunk = pd.read_csv(full_path, sep=sep, chunksize=chunksize, encoding=encoding)
        e_time_chunk = time.time()
        print(f"Data imported from {full_path} in {e_time_chunk - s_time_chunk:.2f} sec")
        return pd.concat(chunk)

    def import_data_excel(self, file_path):
        full_path = self._get_full_path(file_path)
        return pd.read_excel(full_path)

    def import_data_shape(self, file_path):
        full_path = self._get_full_path(file_path)
        return gpd.read_file(full_path)

    def import_frenquency_of_extreme_precipitation_copernicus(self) -> xr.Dataset:
        file_path = "data\\15_frequency_of_extreme_precipitation-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_path, engine="netcdf4")

    def import_maximum_five_days_precipitation_copernicus(self):
        file_path = "data\\13_maximum_five_days_precipitation-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_path, engine="netcdf4")

    def import_extreme_precipitation_total_copernicus(self):
        file_path = "data\\14_extreme_precipitation_total-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_path, engine="netcdf4")

    def import_extreme_wind_speed_days(self):
        file_path = "data\\24_extreme_wind_speed_days-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_path, engine="netcdf4")

    def import_duration_of_meteorological_droughts(self):
        file_paths = "data\\19_duration_of_meteorological_droughts-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")
    
    def import_warmest_three_day_period(self):
        file_paths = "data\\07_warmest_three_day_period-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")

    def import_heat_waves_climatological(self):
        file_paths = "data\\09_heat_waves_climatological-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")

    def import_frost_days(self):
        file_paths ="data\\11_frost_days-reanalysis-yearly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")


data_importer = ImportData()
frenquency_dataset = data_importer.import_frenquency_of_extreme_precipitation_copernicus()
