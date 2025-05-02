import os
import xarray as xr

class ImportData:
    """
    Classe de chargement des données NetCDF ERA5 pressure levels.
    Les fichiers doivent être placés dans le dossier <projet>/data/.
    """

    def __init__(self):
        # chemin absolu vers le dossier du script
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _full_path(self, filename: str) -> str:
        """
        Retourne le chemin complet d'un fichier dans ./data/
        """
        return os.path.join(self.base_path, "data", filename)

    def import_relative_humidity(self) -> xr.Dataset:
        """
        Charge tous les fichiers ERA5 de humidité relative à 1000 hPa.
        Pattern : data/relative_humidity*.nc
        """
        pattern = self._full_path("relative_humidity*.nc")
        return xr.open_mfdataset(pattern, combine="by_coords")

    def import_temperature(self) -> xr.Dataset:
        """
        Charge tous les fichiers ERA5 de température à 1000 hPa.
        Pattern : data/temperature*.nc
        """
        pattern = self._full_path("temperature*.nc")
        return xr.open_mfdataset(pattern, combine="by_coords")

    def import_u_wind(self) -> xr.Dataset:
        """
        Charge tous les fichiers ERA5 de composante U du vent à 1000 hPa.
        Pattern : data/u_component_of_wind*.nc
        """
        pattern = self._full_path("u_component_of_wind*.nc")
        return xr.open_mfdataset(pattern, combine="by_coords")

    def import_v_wind(self) -> xr.Dataset:
        """
        Charge tous les fichiers ERA5 de composante V du vent à 1000 hPa.
        Pattern : data/v_component_of_wind*.nc
        """
        pattern = self._full_path("v_component_of_wind*.nc")
        return xr.open_mfdataset(pattern, combine="by_coords")
