import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np



# Paramètres physiques
longueur_lambda = 300  # en mm
longueur_domaine = 3  # longueur de l'arrêt en mm de l'acinus modélisé en cube 
pas_discretisation = 0.05  # en mm
r_ratio = -longueur_lambda / (pas_discretisation + longueur_lambda)
C1 = 9.28  # Ca-Cb en mmol/l
diffusivite = 19.8  # constante de diffusivité en mm²/s


def discretiser_espace(rectangle, taille_pas):
    # Définir les limites du rectangle principal
    x_min, y_min, x_max, y_max = rectangle

    # Créer une grille discrète
    valeurs_x = list(range(int(x_min), int(x_max) + 1, taille_pas))
    valeurs_y = list(range(int(y_min), int(y_max) + 1, taille_pas))

    # Liste pour stocker les points de la figure
    points_figure = []

    # Ajouter les points du rectangle principal
    for x in valeurs_x:
        for y in valeurs_y:
            points_figure.append((x, y))

    # Trier les points pour un ordre souhaité
    points_figure.sort(key=lambda p: (p[0], p[1]))

    # Dictionnaire pour stocker les voisins de chaque point
    voisins_dict = {}

    # Trouver les voisins de chaque point
    for i, point in enumerate(points_figure, start=1):
        voisins = []

        x, y = point

        # Voisin à gauche
        voisin_gauche = (x - taille_pas, y)
        if voisin_gauche in points_figure:
            voisins.append(points_figure.index(voisin_gauche) + 1)
        else:
            voisins.append(0)

        # Voisin à droite
        voisin_droit = (x + taille_pas, y)
        if voisin_droit in points_figure:
            voisins.append(points_figure.index(voisin_droit) + 1)
        else:
            voisins.append(0)

        # Voisin en dessous
        voisin_bas = (x, y - taille_pas)
        if voisin_bas in points_figure:
            voisins.append(points_figure.index(voisin_bas) + 1)
        else:
            voisins.append(0)

        # Voisin au-dessus
        voisin_haut = (x, y + taille_pas)
        if voisin_haut in points_figure:
            voisins.append(points_figure.index(voisin_haut) + 1)
        else:
            voisins.append(0)

        voisins_dict[i] = voisins

    # Calculer les coefficients pour chaque point
    coefficients_dict = {}
    for i, point in enumerate(points_figure, start=1):
        x, y = point
        k1, k2, k3, k4 = voisins_dict[i]
        voisins_existants = [
            n for n in voisins_dict[i] if n != 0
        ]  # Exclure les voisins inexistants
        coefficients_dict[i] = {
            "k1": 1 if k1 != 0 else 0,
            "k2": 1 if k2 != 0 else 0,
            "k3": 1 if k3 != 0 else 0,
            "k4": 1 if k4 != 0 else 0,
            "k": -len(voisins_existants),
        }

    return points_figure, voisins_dict, coefficients_dict


def assigner_conditions_limites(rectangle, taille_pas):
    """Assigne les conditions limites aux points de la discrétisation."""
    conditions_limites = {}

    # Conditions sur les bords du rectangle principal
    for point in [(0, y) for y in range(rectangle[1], rectangle[3] + 1, taille_pas)]:
        conditions_limites[point] = 1  # Condition de type 1

    for point in [
        (rectangle[2], y) for y in range(rectangle[1], rectangle[3] + 1, taille_pas)
    ]:
        conditions_limites[point] = 2  # Condition de Robin

    return conditions_limites


def construire_matrice_B(
    coefficients, voisins, conditions_limites, points_figure, longueur_lambda
):
    """Construit la matrice B pour le système linéaire."""
    ratio_r = -longueur_lambda / (pas_discretisation + longueur_lambda)
    nombre_points = len(coefficients)

    matrice_B = np.zeros((nombre_points, nombre_points))
    for k in range(1, nombre_points + 1):
        # Soustraire 1 car les indices commencent à 1
        matrice_B[k - 1, voisins[k][0] - 1] = coefficients[k]["k1"]
        matrice_B[k - 1, voisins[k][2] - 1] = coefficients[k]["k3"]
        matrice_B[k - 1, voisins[k][3] - 1] = coefficients[k]["k4"]
        matrice_B[k - 1, voisins[k][1] - 1] = coefficients[k]["k2"]

        matrice_B[k - 1, k - 1] = coefficients[k]["k"]
        # Modifier la ligne k si la condition limite du point k est 1 ou 2
        condition_k = conditions_limites.get(points_figure[k - 1], 0)
        if condition_k == 1:
            matrice_B[k - 1, :] = 0
            matrice_B[k - 1, k - 1] = 1
        if (
            condition_k == 2
            and coefficients[k]["k"] != -4
            and coefficients[k]["k"] != -2
        ):
            matrice_B[k - 1, voisins[k][0] - 1] = ratio_r * (
                coefficients[k]["k1"] * coefficients[k]["k3"] * coefficients[k]["k4"]
            )
            matrice_B[k - 1, voisins[k][1] - 1] = ratio_r * (
                coefficients[k]["k2"] * coefficients[k]["k3"] * coefficients[k]["k4"]
            )
            matrice_B[k - 1, voisins[k][2] - 1] = ratio_r * (
                coefficients[k]["k3"] * coefficients[k]["k2"] * coefficients[k]["k1"]
            )
            matrice_B[k - 1, voisins[k][3] - 1] = ratio_r * (
                coefficients[k]["k4"] * coefficients[k]["k2"] * coefficients[k]["k1"]
            )
            matrice_B[k - 1, k - 1] = 1
        if condition_k == 2 and coefficients[k]["k"] == -2:
            matrice_B[k - 1, voisins[k][0] - 1] = ratio_r * coefficients[k]["k1"]
            matrice_B[k - 1, voisins[k][1] - 1] = ratio_r * coefficients[k]["k2"]
            matrice_B[k - 1, voisins[k][2] - 1] = 0
            matrice_B[k - 1, voisins[k][3] - 1] = 0
            matrice_B[k - 1, k - 1] = 1

    return matrice_B


def tracer_flux(
    points_figure, vecteur_resultat, voisins, taille_pas, diffusivite, intervalle=3
):
    """Trace le champ de flux basé sur les résultats de concentration."""
    # Initialiser les listes pour les coordonnées et les projections des flux
    coord_x, coord_y = zip(*points_figure)
    flux_x = []
    flux_y = []
    norme_flux = []
    x_echantillones = []
    y_echantillones = []

    for i, point in enumerate(points_figure, start=1):
        # Filtrer les points pour ne garder qu'un sous-ensemble (intervalle)
        if i % intervalle == 0:
            deriv_x = 0
            deriv_y = 0
            voisins_point = voisins[i]

            # Dérivée partielle selon x
            if voisins_point[1] != 0:  # Voisin à droite
                deriv_x = (
                    vecteur_resultat[voisins_point[1] - 1] - vecteur_resultat[i - 1]
                ) / pas_discretisation
            elif voisins_point[0] != 0:  # Voisin à gauche
                deriv_x = (
                    vecteur_resultat[i - 1] - vecteur_resultat[voisins_point[0] - 1]
                ) / pas_discretisation

            # Dérivée partielle selon y
            if voisins_point[3] != 0:  # Voisin au-dessus
                deriv_y = (
                    vecteur_resultat[voisins_point[3] - 1] - vecteur_resultat[i - 1]
                ) / pas_discretisation
            elif voisins_point[2] != 0:  # Voisin en dessous
                deriv_y = (
                    vecteur_resultat[i - 1] - vecteur_resultat[voisins_point[2] - 1]
                ) / pas_discretisation

            # Calcul du flux
            flux_x_val = -diffusivite * deriv_x
            flux_y_val = -diffusivite * deriv_y

            flux_x.append(flux_x_val)
            flux_y.append(flux_y_val)
            flux_norm = np.sqrt(flux_x_val**2 + flux_y_val**2)
            norme_flux.append(flux_norm)

            x_echantillones.append(point[0])
            y_echantillones.append(point[1])

    # Tracer le champ de flux
    plt.figure(figsize=(8, 8))

    # Calculer l'échelle en fonction du pas de discretisation

    plt.quiver(
        x_echantillones,
        y_echantillones,
        flux_x,
        flux_y,
        angles="xy",
        scale_units="xy",
        scale=0.18,
        cmap="viridis",
        color="blue",
    )

    plt.title(f"Champ de flux (-D.grad(C)) à {flux_norm} mmol/(m²s)")
    plt.xlabel("x (10^-2 mm)")
    plt.ylabel("y (10^-2 mm)")
    plt.grid(True)
    plt.show()


def valider_solution_physique(
    pts_figure, vecteur_resultat, C1, longueur_lambda, longueur_domaine
):

    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Extraire les points de la première ligne (y=0)
    valeurs_x = []
    concentration_numerique_x = []

    for i, point in enumerate(pts_figure):
        if point[1] == 0:  # Ligne avec y=0
            valeurs_x.append(point[0])
            concentration_numerique_x.append(
                vecteur_resultat[i]
            )  # Concentration correspondante

    # Calculer la solution analytique pour y=0
    concentration_analytique_x = [
        -C1 / (longueur_lambda + longueur_domaine) * x / 100 + C1 for x in valeurs_x
    ]

    # Premier graphique : Variation de concentration
    ax1.plot(
        valeurs_x,
        concentration_numerique_x,
        label="Solution numérique (C vs x, y=0)",
        marker="o",
    )
    ax1.plot(
        valeurs_x,
        concentration_analytique_x,
        label="Solution analytique (C vs x, y=0)",
        linestyle="--",
    )
    ax1.set_title("Variation de C suivant x (y=0)")
    ax1.set_xlabel("Position x (mm)")
    ax1.set_ylabel("Concentration (mmol/L)")
    ax1.grid(True)
    ax1.legend()

    # Calculer l'erreur relative
    erreur_relative = [
        abs((num - ana) / ana) * 100 if ana != 0 else 0
        for num, ana in zip(concentration_numerique_x, concentration_analytique_x)
    ]

    # Deuxième graphique : Erreur relative
    ax2.plot(
        valeurs_x, erreur_relative, label="Erreur relative", marker="o", color="red"
    )
    ax2.set_title("Erreur relative entre solution numérique et analytique")
    ax2.set_xlabel("Position x (mm)")
    ax2.set_ylabel("Erreur relative (%)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Optionnel : retourner les valeurs d'erreur si nécessaire
    return {"erreur_relative": erreur_relative}


def tracer_points_conditions(coord_x, coord_y, conditions):
    """Trace les points avec leurs conditions limites."""
    couleurs = [conditions.get((x, y), 0) for x, y in zip(coord_x, coord_y)]
    plt.scatter(coord_x, coord_y, c=couleurs, cmap="plasma")
    plt.title("Discrétisation de l'espace avec les conditions aux limites")
    plt.xlabel("Position x (en 10^-2 mm)")
    plt.ylabel("Position y (en 10^-2 mm)")
    plt.grid(True)
    plt.show()


def tracer_carte_concentration(pts_figure, valeurs):
    """Trace une carte de concentration basée sur les valeurs calculées."""
    coord_x, coord_y = zip(*pts_figure)
    unique_x = sorted(set(coord_x))
    unique_y = sorted(set(coord_y))

    # Créer une grille pour la carte de chaleur
    carte_chaleur = np.zeros((len(unique_y), len(unique_x)))

    # Remplir la grille avec les valeurs
    for point, valeur in zip(pts_figure, valeurs):
        x, y = point
        i = unique_y.index(y)
        j = unique_x.index(x)
        carte_chaleur[i, j] = valeur

    # Définir la colormap personnalisée
    cmap_inferno = plt.cm.inferno
    couleurs_custom = ["green"] + list(cmap_inferno(np.linspace(0, 1, 1000)))
    cmap_personnalisee = ListedColormap(couleurs_custom)

    # Tracer l'image
    plt.imshow(
        carte_chaleur,
        cmap="inferno",
        extent=[min(unique_x), max(unique_x), min(unique_y), max(unique_y)],
        origin="lower",
        vmin=np.min(valeurs),
        vmax=np.max(valeurs),
    )
    plt.colorbar(label="Concentration en mmol/l")
    plt.title("Concentration en O2")
    plt.xlabel("Longueur (en 10^-2 mm)")
    plt.ylabel("Longueur (en 10^-2 mm)")
    plt.show()


def tracer_concentration_vs_lambda(
    pts_figure,
    conditions_limites,
    voisins,
    coefficients,
    C1,
    longueur_domaine,
    taille_pas,
):

    # Plage de valeurs de lambda à explorer
    valeurs_lambda = np.linspace(1, 500, 30)

    # Listes pour stocker les concentrations
    concentrations_mur_numerique = []
    concentrations_mur_analytique = []

    for lambda_val in valeurs_lambda:
        # Solution numérique
        matrice_B = construire_matrice_B(
            coefficients, voisins, conditions_limites, pts_figure, lambda_val
        )

        # Inverser la matrice B
        inverse_matrice_B = np.linalg.inv(matrice_B)

        # Préparer le vecteur des conditions aux limites
        vecteur_conditions = np.zeros(len(pts_figure))

        # Appliquer les conditions aux limites
        for i, point in enumerate(pts_figure, start=1):
            if conditions_limites.get(point, 0) == 1:
                vecteur_conditions[i - 1] = C1
            elif conditions_limites.get(point, 0) == 2:
                vecteur_conditions[i - 1] = 0

        # Résoudre pour les concentrations
        vecteur_resultat = np.dot(inverse_matrice_B, vecteur_conditions)

        # Trouver les concentrations à la paroi droite (x = max)
        max_x = max(p[0] for p in pts_figure)
        points_mur = [
            vecteur_resultat[i]
            for i, point in enumerate(pts_figure)
            if point[0] == max_x
        ]

        # Moyenne des concentrations à la paroi (numérique)
        concentration_mur_numerique = np.mean(points_mur)
        concentrations_mur_numerique.append(concentration_mur_numerique)

        # Solution analytique
        concentration_analytique = (
            -C1 / (lambda_val + longueur_domaine) * 300 / 100 + C1
        )
        concentrations_mur_analytique.append(concentration_analytique)

    # Tracer le graphique
    plt.figure(figsize=(12, 7))

    # Tracer les solutions numériques
    plt.plot(
        valeurs_lambda,
        concentrations_mur_numerique,
        marker="o",
        label="Solution Numérique",
        color="blue",
    )

    # Tracer les solutions analytiques
    plt.plot(
        valeurs_lambda,
        concentrations_mur_analytique,
        marker="x",
        label="Solution Analytique",
        color="red",
        linestyle="--",
    )

    plt.title("Concentration à la paroi droite en fonction de λ")
    plt.xlabel("λ (mm)")
    plt.ylabel("Concentration à la paroi droite (mmol/L)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "valeurs_lambda": valeurs_lambda,
        "concentrations_numeriques": concentrations_mur_numerique,
        "concentrations_analytiques": concentrations_mur_analytique,
    }




def main():
    rectangle_principal = (0, 0, 300, 300)
    taille_pas = int(pas_discretisation * 100)
    

    # Discrétiser l'espace
    points_discrets, voisins, coefficients = discretiser_espace(
        rectangle_principal, taille_pas
    )
    conditions_limites = assigner_conditions_limites(rectangle_principal, taille_pas)

    # Valeur de lambda fixée
    longueur_lambda_fixe = longueur_lambda
    matrice_B = construire_matrice_B(
        coefficients, voisins, conditions_limites, points_discrets, longueur_lambda_fixe
    )

    inverse_matrice_B = np.linalg.inv(matrice_B)
    vecteur_conditions = np.zeros(len(points_discrets))

    # Initialiser les variables pour le calcul du débit
    concentration_initiale = C1

    for i, point in enumerate(points_discrets, start=1):
        if conditions_limites.get(point, 0) == 1:
            vecteur_conditions[i - 1] = concentration_initiale
        if conditions_limites.get(point, 0) == 2:
            vecteur_conditions[i - 1] = 0

    # Résoudre le système pour les concentrations
    vecteur_resultat = np.dot(inverse_matrice_B, vecteur_conditions)

    coord_x, coord_y = zip(*points_discrets)
    tracer_points_conditions(coord_x, coord_y, conditions_limites)
    tracer_carte_concentration(points_discrets, vecteur_resultat)

    valider_solution_physique(
        points_discrets,
        vecteur_resultat,
        C1=concentration_initiale,
        longueur_lambda=longueur_lambda_fixe,
        longueur_domaine=3,
    )

    tracer_flux(
        points_discrets,
        vecteur_resultat,
        voisins,
        taille_pas,
        diffusivite,
        intervalle=1,
    )

    # Définir les valeurs fixes pour les graphiques de débit

    C1_fixe = concentration_initiale
    debit_total_entree = 0
    debit_total_sortie = 0
    nombre_points = len(points_discrets)
    c = C1_fixe

    for k in range(1, nombre_points + 1):
        condition_k = conditions_limites.get(points_discrets[k - 1], 0)
        if condition_k == 2:
            debit_total_entree += (
                vecteur_resultat[k - 1]
                * diffusivite
                * pas_discretisation
                * 22.4
                * longueur_domaine
                * 1e-6
                / longueur_lambda_fixe
            )
        if condition_k == 1:
            debit_total_sortie += (
                (vecteur_resultat[voisins[k][1] - 1] - vecteur_resultat[k - 1])
                * diffusivite
                * pas_discretisation
                * 22.4
                * longueur_domaine
                / pas_discretisation
                * 1e-6
            )

    # Calcul du débit pour la figure considérée par l'utilisateur

    debit_vs_lambda = []  # Pour lambda

    # Définir les valeurs à explorer
    valeurs_lambda = np.linspace(
        100, 400, 20
    )  # Variation de lambda entre 100 et 400 mm

    # Boucle pour le graphe en fonction de C1 (lambda fixé)

    # Boucle pour le graphe en fonction de lambda (C1 fixé)
    for lambda_val in valeurs_lambda:
        # Mettre à jour les valeurs des conditions limites pour le C1 fixé
        matrice_B = construire_matrice_B(
            coefficients, voisins, conditions_limites, points_discrets, lambda_val
        )

        inverse_matrice_B = np.linalg.inv(matrice_B)
        vecteur_conditions = np.zeros(len(points_discrets))

        # Identifier les points avec condition limite 2
        for i, point in enumerate(points_discrets, start=1):
            if conditions_limites.get(point, 0) == 1:
                vecteur_conditions[i - 1] = C1_fixe
            if conditions_limites.get(point, 0) == 2:
                vecteur_conditions[i - 1] = 0  # C2 reste constant

        vecteur_resultat = np.dot(inverse_matrice_B, vecteur_conditions)
        # Calculer le débit total pour le lambda actuel
        debit_total = 0
        for k in range(1, len(points_discrets) + 1):
            condition_k = conditions_limites.get(points_discrets[k - 1], 0)
            if condition_k == 1:
                debit_total -= (
                    (vecteur_resultat[voisins[k][1] - 1] - vecteur_resultat[k - 1])
                    * diffusivite
                    * pas_discretisation
                    * 22.4
                    * longueur_domaine
                    / pas_discretisation
                ) * 1e-6  # Robin

        # Ajouter les résultats au graphe
        debit_vs_lambda.append((np.log(lambda_val), np.log(debit_total)))

    # Tracer les graphiques de débit

    # 1. Graphique : Débit en fonction de lambda
    debit_vs_lambda = np.array(debit_vs_lambda)
    plt.figure(figsize=(8, 6))
    plt.plot(
        debit_vs_lambda[:, 0],
        debit_vs_lambda[:, 1],
        marker="o",
        label=f"C1 fixé = {C1_fixe} mmol/l",
    )
    plt.title("Débit Volumique en fonction de Lambda")
    plt.xlabel("ln(Lambda) (mm)")
    plt.ylabel("ln(Débit) (ml/s)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Tracer la variation de concentration en fonction de lambda
    tracer_concentration_vs_lambda(
        points_discrets,
        conditions_limites,
        voisins,
        coefficients,
        C1=9.28,  # Concentration initiale
        longueur_domaine=3,  # Longueur du domaine
        taille_pas=5,  # Taille du pas
    )
    
    print(
        "Valeur du débit correspondant à la figure choisie en ml/s (entrée et sortie)"
    )
    print("debit entrant ", abs(debit_total_entree))
    print("debit sortant ", abs(debit_total_sortie))


if __name__ == "__main__":
    main()
