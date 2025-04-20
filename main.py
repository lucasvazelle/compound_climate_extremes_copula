from import_data import ImportData
import numpy as np
import matplotlib.pyplot as plt
from copulas.bivariate import Gumbel
from typing import Tuple
import xarray as xr
from scipy.stats import norm
import seaborn as sns
from copulae import pseudo_obs
import pandas as pd
import copulas.bivariate
import pandas as pd
import netCDF4 
import os

class AnalyseurCopule:
    def __init__(
        self,
        importeur1: xr.Dataset,
        importeur2: xr.Dataset,
        nom_variable1,
        nom_variable2,
        lat: float,
        lon: float,
    ) -> None:
        self.importeur1: xr.Dataset = importeur1
        self.importeur2: xr.Dataset = importeur2
        self.nom_variable1: str = nom_variable1
        self.nom_variable2: str = nom_variable2
        self.lat: float = lat
        self.lon: float = lon
        self.donnees1: np.ndarray = None
        self.donnees2: np.ndarray = None
        self.donnees_normalise_1: np.ndarray = None
        self.donnees_normalise_2: np.ndarray = None
        self.theta: float = None
        self.copule_selectionnee = None
        self.nom_copule = None


    def charger_donnees(self) -> None:
        self.donnees1 = self.importeur1[list(self.importeur1.data_vars.keys())[0]][
            :, self.lat, self.lon
        ]
        self.donnees2 = self.importeur2[list(self.importeur2.data_vars.keys())[0]][
            :, self.lat, self.lon
        ]

    def graph_donnees_multivariees(self) -> plt.savefig:
        h = sns.jointplot(x=self.donnees1, y=self.donnees2, kind="scatter")
        h.set_axis_labels(f"{self.nom_variable1}", f"{self.nom_variable2}", fontsize=10)
        return h.savefig(f"output/distribution_jointe_{self.nom_variable1}_{self.nom_variable2}.png")

    @staticmethod
    def normaliser_rank(donnees: np.ndarray) -> np.ndarray:
        return pseudo_obs(donnees)

    def normaliser_donnees(self) -> None:
        self.donnees_normalise_1 = self.normaliser_rank(self.donnees1)
        self.donnees_normalise_2 = self.normaliser_rank(self.donnees2)

    def graph_donnees_multivariees__normalisees(self) -> plt.savefig:
        h2 = sns.jointplot(
            x=self.donnees_normalise_1, y=self.donnees_normalise_2, kind="scatter"
        )
        h2.set_axis_labels(
            f"{self.nom_variable1}", f"{self.nom_variable2}", fontsize=10
        )
        return h2.savefig(f"output/distribution_jointe_normalisé_{self.nom_variable1}_{self.nom_variable2}.png")

    def verifie_type_de_copule(self):
        npdata = pd.DataFrame(
            {"X": self.donnees_normalise_1, "Y": self.donnees_normalise_2}
        ).to_numpy()
        result = copulas.bivariate.base.Bivariate().select_copula(npdata)
        self.copule_selectionnee = result
        self.nom_copule = result.to_dict()["copula_type"]

    def calculer_theta(self) -> float:
        if hasattr(self.copule_selectionnee, 'theta'):
            self.theta = self.copule_selectionnee.theta
            return self.theta
        else:
            self.theta = None
            return None

    def bootstrap_theta(self, n_iterations: int = 1000) -> np.ndarray:
        if not hasattr(self.copule_selectionnee, 'theta'):
            return np.array([None] * n_iterations)

        bootstrap_thetas = []
        n = len(self.donnees_normalise_1)
        for _ in range(n_iterations):
            indices = np.random.randint(0, n, n)
            donnees_bootstrap = np.column_stack((
                self.donnees_normalise_1[indices],
                self.donnees_normalise_2[indices]
            ))
            copule = type(self.copule_selectionnee)()
            copule.fit(donnees_bootstrap)
            bootstrap_thetas.append(copule.theta)
        return np.array(bootstrap_thetas)

    def test_significatif_theta(self, niveau_confiance: float = 0.95) -> bool:
        # Bootstrap pour estimer l'erreur standard de theta
        bootstrap_thetas = self.bootstrap_theta()
        erreur_standard = np.std(bootstrap_thetas)
        # Estimation de l'intervalle de confiance pour theta
        z = norm.ppf(1 - (1 - niveau_confiance) / 2)
        self.theta_ci = (
            self.theta - z * erreur_standard,
            self.theta + z * erreur_standard,
        )
        # Test si theta est significativement supérieur à 1
        return self.theta_ci[0] > 1

    def tau_de_kendal(self) -> float:
        if self.theta:
            return self.theta / (self.theta + 2)
        else:
            return np.nan

    def enregistrer_graph_distribution_jointe_et_copule(self) -> plt.savefig:
        donnees = np.column_stack(
            (self.donnees_normalise_1, self.donnees_normalise_2)
        )
        self.copule_selectionnee.fit(donnees)
        u = np.linspace(0, 1, 100)
        v = np.linspace(0, 1, 100)
        U, V = np.meshgrid(u, v)
        Z = self.copule_selectionnee.cumulative_distribution(
            np.column_stack((U.ravel(), V.ravel()))
        ).reshape(100, 100)

        plt.figure(figsize=(10, 6))
        contour = plt.contour(U, V, Z, levels=np.linspace(0, 1, 11), cmap="Blues")
        cbar = plt.colorbar(contour)
        cbar.set_label("Courbes de niveau")

        plt.scatter(
            self.donnees_normalise_1,
            self.donnees_normalise_2,
            c="red",
            label="Observations normalisées",
            alpha=0.5,
        )# Déterminer le contenu du titre dynamiquement
        tau_kendall = self.tau_de_kendal()
        theta_display = f"{round(self.theta, 2)}" if self.theta is not None else "N/A"
        tau_display = f"{round(tau_kendall, 2)}" if not np.isnan(tau_kendall) else "N/A"

        titre = (
            f"Copule sélectionnée : {self.nom_copule} | "
            f"Paramètre θ = {theta_display} | "
            f"Tau de Kendall = {tau_display}"
        )

        plt.title(titre)
        plt.xlabel(f"Valeurs normalisées de {self.nom_variable1}")
        plt.ylabel(f"Valeurs normalisées de {self.nom_variable2}")
        plt.legend()
        return plt.savefig(f"output/Copule_{self.nom_copule}_{self.nom_variable1}_{self.nom_variable2}.png")

    def lancer_analyse_copule(self):
        self.charger_donnees()
        self.graph_donnees_multivariees()
        self.normaliser_donnees()
        self.graph_donnees_multivariees__normalisees()
        self.verifie_type_de_copule()
        self.calculer_theta()
        self.tau_de_kendal()
        self.enregistrer_graph_distribution_jointe_et_copule()
        return (
            self.calculer_theta(),
            self.tau_de_kendal(),
            self.test_significatif_theta(),
            self.theta_ci,
        )


# _____________________________________paramètres_______________________________________
importeur1 = ImportData().import_extreme_wind_speed_days()
importeur2 = ImportData().import_heat_waves_climatological()
nom_variable1 = "Vents extrêmes"
nom_variable2 = "Vagues de chaleur"
lat, lon = 100, 65
# _____________________________________lance la classe_____________________________
analyseur = AnalyseurCopule(
    importeur1, importeur2, nom_variable1, nom_variable2, lat, lon
)
theta, tau_kendal, significatif, theta_ci = analyseur.lancer_analyse_copule()
print(f"Theta: {theta}")
print(f"Theta est significativement supérieur à 1: {significatif}")
print(f"Intervalle de confiance pour theta: {theta_ci}")
print(f"Tau de Kendall: {tau_kendal}")