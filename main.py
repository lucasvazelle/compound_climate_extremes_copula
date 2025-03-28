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
        return h.savefig("distribution_jointe.png")

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
        return h2.savefig("distribution_jointe_normalisé.png")

    def verifie_type_de_copule(self):
        npdata = pd.DataFrame(
            {"X": self.donnees_normalise_1, "Y": self.donnees_normalise_2}
        ).to_numpy()
        result = copulas.bivariate.base.Bivariate().select_copula(npdata)
        parameters = result.to_dict()
        copule = parameters["copula_type"]
        assert copule == "GUMBEL"
        return None

    def calculer_theta_gumbel(self) -> float:
        copule_gumbel: Gumbel = Gumbel()
        donnees: np.ndarray = np.column_stack(
            (self.donnees_normalise_1, self.donnees_normalise_2)
        )
        copule_gumbel.fit(donnees)
        self.theta = copule_gumbel.theta
        return self.theta

    def bootstrap_theta(self, n_iterations: int = 1000) -> np.ndarray:
        bootstrap_thetas = []
        n = len(self.donnees_normalise_1)
        for _ in range(n_iterations):
            indices = np.random.randint(0, n, n)
            donnees_bootstrap1 = self.donnees_normalise_1[indices]
            donnees_bootstrap2 = self.donnees_normalise_2[indices]
            copule_gumbel: Gumbel = Gumbel()
            donnees_bootstrap = np.column_stack(
                (donnees_bootstrap1, donnees_bootstrap2)
            )
            copule_gumbel.fit(donnees_bootstrap)
            bootstrap_thetas.append(copule_gumbel.theta)
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
        theta = self.calculer_theta_gumbel()
        tau_kendall = theta / (theta + 2)
        return tau_kendall

    def enregistrer_graph_distribution_jointe_et_copule(self) -> plt.savefig:
        copule_gumbel: Gumbel = Gumbel()
        donnees: np.ndarray = np.column_stack(
            (self.donnees_normalise_1, self.donnees_normalise_2)
        )
        copule_gumbel.fit(donnees)
        u: np.ndarray = np.linspace(0, 1, 100)
        v: np.ndarray = np.linspace(0, 1, 100)
        U, V = np.meshgrid(u, v)
        Z: np.ndarray = copule_gumbel.cumulative_distribution(
            np.column_stack((U.ravel(), V.ravel()))
        ).reshape(100, 100)
        plt.figure(figsize=(10, 6))
        contour = plt.contour(U, V, Z, levels=np.linspace(0, 1, 11), cmap="Blues")
        # plt.colorbar(contour)
        cbar = plt.colorbar(contour)
        cbar.set_label("Courbes de niveau")

        plt.scatter(
            self.donnees_normalise_1,
            self.donnees_normalise_2,
            c="red",
            label="Observations normalisées",
            alpha=0.5,
        )
        plt.title(
            f"Copule - La relation de dépendance est donnée par la valeur rho = {round(self.calculer_theta_gumbel(),1)}"
        )
        plt.xlabel(f"Valeurs normalisées de {self.nom_variable1}")
        plt.ylabel(f"Valeurs normalisées de {self.nom_variable2}")
        plt.legend()
        return plt.savefig(f"Copule de Gumbel.png")

    def lancer_analyse_copule(self):
        self.charger_donnees()
        self.graph_donnees_multivariees()
        self.normaliser_donnees()
        self.graph_donnees_multivariees__normalisees()
        self.verifie_type_de_copule()
        self.enregistrer_graph_distribution_jointe_et_copule()
        return (
            self.calculer_theta_gumbel(),
            self.tau_de_kendal(),
            self.test_significatif_theta(),
            self.theta_ci,
        )


# _____________________________________paramètres_______________________________________
importeur1 = ImportData().import_extreme_wind_speed_days()
importeur2 = ImportData().import_frenquency_of_extreme_precipitation_copernicus()
nom_variable1 = "Total des précipitations sur les 5 jours consécutifs les plus pluvieux"
nom_variable2 = "Total sur une année des précipitations extrêmes"
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