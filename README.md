# compound_climate_extremes_copula

Ce projet permet de mesurer des dépendances bivariés de la chaleur, du temps sec et des vents extrêmes partout en Europe.
La modélisation de la structure de dépendance se fait avec la méthode des copules (voir fichier.pptx.pdf). 

**Etape préliminaire**

Installer python 3.10 ou 3.11

**Etape 1 Installer les dépendances**

pip install -r requierments.txt

**Modifier main.py selon vos paramètres**

lat_sel, lon_sel = 50, 2 # chosir respectivement la latitude et longitude
ds1 = dry                       # choisir parmi hot, dry et wind
ds2 = hot                       # choisir parmi hot, dry et wind
nom_de_la_variable_1 = "dry %"    # à adapter
nom_de_la_variable_2 = "hot degrés"   # à adapter


**Lancer l'analyse**
$python main.py

Les résultats apparaitrons dans output/ et dans le log