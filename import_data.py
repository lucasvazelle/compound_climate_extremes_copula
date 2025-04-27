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

    def import_wind(self):
        file_path = "data\\24_extreme_wind_speed_days-reanalysis-monthly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_path, engine="netcdf4")

    def import_dry(self):
        file_paths = "data\\18_consecutive_dry_days-reanalysis-monthly-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")
    
    def import_hot_35deg(self):
        file_paths = "data\\06_hot_days-reanalysis-monthly-35deg-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")

    def import_hot_30deg(self):
        file_paths = "data\\06_hot_days-reanalysis-monthly-30deg-grid-1940-2023-v1.0.nc"
        return xr.open_dataset(file_paths, engine="netcdf4")


data_importer = ImportData()
test = data_importer.import_hot_30deg()