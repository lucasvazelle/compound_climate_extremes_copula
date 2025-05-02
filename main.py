import numpy as np
import xarray as xr
from import_data import ImportData
from copule_bivarie import AnalyseurCopule  
from transform_data import transformation_unite_de_mesure_data

if __name__ == "__main__":
    importer = ImportData()
    dry, hot, wind = transformation_unite_de_mesure_data(importer)

    #________________paramètres___________________________
    lat_sel, lon_sel = 30, 0 # chosir
    ds1 = dry                       # choisir parmi hot, dry et wind
    ds2 = hot                       # choisir parmi hot, dry et wind
    nom_de_la_variable_1 = "dry %"    # à adapter
    nom_de_la_variable_2 = "hot degrés"   # à adapter
    #________________paramètres___________________________

    analyseur = AnalyseurCopule(ds1, ds2, nom_de_la_variable_1, nom_de_la_variable_2, lat_sel, lon_sel)
    theta, tau, significant, ci = analyseur.lancer_analyse_copule()
    print(f"Copule {analyseur.nom_copule} | θ={theta:.2f} | τ={tau:.2f} | CI={ci}")
