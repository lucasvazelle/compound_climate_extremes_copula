# compound_climate_extremes_copula

Mesure des dépendances de la chaleur, du temps sec et des vents extrêmes partout en Europe.
Modélisation de la structure de dépendance grâce aux copules.

** Installer les dépendances**

pip install -r requierments.txt

** modifier main.py selon vos paramètres**

importeur1 = ImportData().import_dry() # choisir les données que vous voulez disponible dans import_data.py
importeur2 = ImportData().import_wind() # choisir les données que vous voulez disponible dans import_data.py
nom_variable1 = "dry" # adaptez selon la donnée choisie
nom_variable2 = "wind" # adaptez selon la donnée choisie
lat, lon = 26, 36  # choisir une localisation

** Lancer l'analyse **
python main.py

Les résultats apparaitrons dans output/ et dans le log