import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.widgets import Button, Slider
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import tkinter as tk
from tkinter import messagebox
import math

# Liste pour stocker les conditions limites de Robin
bordures_Robin = []

# Paramètres physiques
longueur_lambda = 300  # en mm
pas_discretisation = 0.05  # en mm
r_ratio = -longueur_lambda / (pas_discretisation + longueur_lambda)
C1 = 9.28  # Ca-Cb en mmol/l
longueur_domaine = 3  # longueur de l'arrêt de l'acinus modélisé en cube
diffusivite = 19.8  # constante de diffusivité en mm²/s


class EditeurFormes:
    def __init__(self, axe, rectangle, taille_pas):
        self.axe = axe
        self.compteur = 0
        self.rectangle_principal = rectangle
        self.taille_pas = taille_pas
        self.sous_rectangles = []
        self.sous_triangles = []
        self.sous_cercles = []
        self.sous_polygones = []  # Liste pour stocker les formes
        self.forme_actuelle = None  # Suivre la forme en cours de dessin
        self.axe.set_xlim(rectangle[0] - 5, rectangle[2] + 5)
        self.axe.set_ylim(rectangle[1] - 5, rectangle[3] + 5)

        self.axe.set_title("Cliquez pour définir une forme")

        self.axe.figure.canvas.mpl_connect("button_press_event", self.clic_souris)

    def clic_souris(self, evenement):
        if evenement.inaxes != self.axe:
            return
        largeur, hauteur = curseur_largeur.val, curseur_hauteur.val
        self.x, self.y = evenement.xdata, evenement.ydata
        self.compteur += 1

        if self.type_forme == "rectangle":
            largeur, hauteur = curseur_largeur.val, curseur_hauteur.val
            rect = Rectangle(
                (self.x, self.y), largeur, hauteur, fill=False, edgecolor="blue"
            )
            self.axe.add_patch(rect)
            self.forme_actuelle = rect
            self.point1 = (
                int(self.x / self.taille_pas) * self.taille_pas,
                int(self.y / self.taille_pas) * self.taille_pas,
            )
            self.point2 = (
                int((self.x + largeur) / self.taille_pas) * self.taille_pas,
                int((self.y + hauteur) / self.taille_pas) * self.taille_pas,
            )

        elif self.type_forme == "cercle":
            rayon = curseur_largeur.val
            circ = Ellipse(
                (self.x, self.y),
                width=rayon * 2,
                height=rayon * 2,
                fill=False,
                edgecolor="green",
            )
            self.axe.add_patch(circ)
            self.forme_actuelle = circ
            self.centre = (
                int(self.x / self.taille_pas) * self.taille_pas,
                int(self.y / self.taille_pas) * self.taille_pas,
            )
            self.rayon1 = curseur_largeur.val
            self.rayon2 = curseur_hauteur.val

        elif self.type_forme == "polygone":
            # Créer un triangle droit
            poly = Polygon(
                [
                    [self.x, self.y],
                    [self.x - curseur_largeur.val, self.y],
                    [self.x, self.y + curseur_hauteur.val],
                ],
                fill=False,
                edgecolor="orange",
            )
            self.axe.add_patch(poly)
            self.forme_actuelle = poly
            self.point4 = (
                int(self.x / self.taille_pas) * self.taille_pas,
                int(self.y / self.taille_pas) * self.taille_pas,
            )
            self.point5 = (
                int((self.x - curseur_largeur.val) / self.taille_pas) * self.taille_pas,
                int(self.y / self.taille_pas) * self.taille_pas,
            )
            self.point6 = (
                int(self.x / self.taille_pas) * self.taille_pas,
                int((self.y + curseur_hauteur.val) / self.taille_pas) * self.taille_pas,
            )

        # Redessiner la figure
        self.axe.figure.canvas.draw()

    def definir_type_forme(self, type_forme):
        """Définit le type de forme à créer (rectangle, cercle, polygone)."""
        self.type_forme = type_forme

    def ajouter_sous_rectangle(self, point_debut, point_fin):
        x_min, y_min = min(point_debut[0], point_fin[0]), min(
            point_debut[1], point_fin[1]
        )
        x_max, y_max = max(point_debut[0], point_fin[0]), max(
            point_debut[1], point_fin[1]
        )

        sous_rect = (x_min, y_min, x_max, y_max)
        self.sous_rectangles.append(sous_rect)

    def ajouter_sous_polygone(self, point_a, point_b, point_c):
        sous_polygone = (point_a, point_b, point_c)
        self.sous_polygones.append(sous_polygone)

    def ajouter_sous_cercle(self, centre, rayon1, rayon2):
        sous_cercle = (centre[0], centre[1], rayon1, rayon2)
        self.sous_cercles.append(sous_cercle)

    def mettre_a_jour_taille_forme(self, valeur):
        """Met à jour la taille de la forme actuelle en fonction des valeurs des curseurs."""
        largeur, hauteur = curseur_largeur.val, curseur_hauteur.val
        if self.forme_actuelle:
            if isinstance(self.forme_actuelle, Rectangle):
                self.forme_actuelle.set_width(curseur_largeur.val)
                self.forme_actuelle.set_height(curseur_hauteur.val)
                self.point1 = (
                    int(self.x / self.taille_pas) * self.taille_pas,
                    int(self.y / self.taille_pas) * self.taille_pas,
                )
                self.point2 = (
                    int((self.x + largeur) / self.taille_pas) * self.taille_pas,
                    int((self.y + hauteur) / self.taille_pas) * self.taille_pas,
                )

            elif isinstance(self.forme_actuelle, Ellipse):
                self.forme_actuelle.width = curseur_largeur.val
                self.forme_actuelle.height = curseur_hauteur.val
                self.centre = (
                    int(self.x / self.taille_pas) * self.taille_pas,
                    int(self.y / self.taille_pas) * self.taille_pas,
                )
                self.rayon1 = float(largeur / 2)
                self.rayon2 = float(hauteur / 2)

            elif isinstance(self.forme_actuelle, Polygon):
                # Mettre à jour la taille du triangle en ajustant ses sommets
                x, y = self.forme_actuelle.xy[0]  # Coin inférieur gauche
                largeur = curseur_largeur.val
                hauteur = curseur_hauteur.val

                # Mettre à jour les sommets du triangle
                self.forme_actuelle.xy = [[x, y], [x - largeur, y], [x, y + hauteur]]

                # Mettre à jour les points du triangle
                self.point4 = (
                    int(x / self.taille_pas) * self.taille_pas,
                    int(y / self.taille_pas) * self.taille_pas,
                )
                self.point5 = (
                    int((x - largeur) / self.taille_pas) * self.taille_pas,
                    int(y / self.taille_pas) * self.taille_pas,
                )
                self.point6 = (
                    int(x / self.taille_pas) * self.taille_pas,
                    int((y + hauteur) / self.taille_pas) * self.taille_pas,
                )

            self.axe.figure.canvas.draw()

    def ajouter_condition_Robin(self):
        if self.type_forme == "rectangle":
            x_min_sub, y_min_sub, x_max_sub, y_max_sub = (
                self.point1[0] - self.taille_pas,
                self.point1[1] - self.taille_pas,
                self.point2[0] + self.taille_pas,
                self.point2[1] + self.taille_pas,
            )
            print(x_min_sub, y_min_sub, x_max_sub, y_max_sub)

            fenetre = (
                tk.Toplevel()
            )  # Utiliser Toplevel pour éviter plusieurs fenêtres racines
            fenetre.title("Conditions Limites de Robin")
            fenetre.grab_set()  # Rendre la fenêtre modale

            # Variables pour les cases à cocher
            var_gauche = tk.BooleanVar(value=False)
            var_droite = tk.BooleanVar(value=False)
            var_bas = tk.BooleanVar(value=False)
            var_haut = tk.BooleanVar(value=False)

            # Fonction de soumission
            def soumettre():
                for y in range(y_min_sub, y_max_sub + 1, self.taille_pas):
                    if var_gauche.get():
                        bordures_Robin.append((x_min_sub, y))  # Ajouter bord gauche
                    if var_droite.get():
                        bordures_Robin.append((x_max_sub, y))  # Ajouter bord droit

                for x in range(x_min_sub, x_max_sub + 1, self.taille_pas):
                    if var_bas.get():
                        bordures_Robin.append((x, y_min_sub))  # Ajouter bord bas
                    if var_haut.get():
                        bordures_Robin.append((x, y_max_sub))  # Ajouter bord haut

                messagebox.showinfo("Information", "Conditions limites ajoutées.")
                fenetre.destroy()  # Fermer la fenêtre

            # Création des cases à cocher
            tk.Checkbutton(
                fenetre, text="Pour le côté gauche", variable=var_gauche
            ).pack(anchor="w")
            tk.Checkbutton(
                fenetre, text="Pour le côté droit", variable=var_droite
            ).pack(anchor="w")
            tk.Checkbutton(fenetre, text="Pour le côté bas", variable=var_bas).pack(
                anchor="w"
            )
            tk.Checkbutton(fenetre, text="Pour le côté haut", variable=var_haut).pack(
                anchor="w"
            )

            # Bouton de soumission
            tk.Button(fenetre, text="Soumettre", command=soumettre).pack()

            # Attendre la fermeture de la fenêtre
            fenetre.wait_window(fenetre)

            print("Rectangle ajouté")

        if self.type_forme == "polygone":
            # Définir les points du triangle
            PointA, PointB, PointC = self.point4, self.point5, self.point6
            arêtes = [(PointA, PointB), (PointB, PointC), (PointC, PointA)]
            points_triangle = [PointA, PointB, PointC]
            for coin in points_triangle:
                x_coin = round(coin[0] / self.taille_pas) * self.taille_pas
                y_coin = round(coin[1] / self.taille_pas) * self.taille_pas
                bordures_Robin.append((x_coin, y_coin))
            for arête in arêtes:
                p1, p2 = arête
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                nombre_points = int(distance // self.taille_pas) + 1

                for i in range(nombre_points + 1):
                    x = p1[0] + i * (p2[0] - p1[0]) / nombre_points
                    y = p1[1] + i * (p2[1] - p1[1]) / nombre_points

                    # Arrondir au point de grille le plus proche
                    x_grille = round(x / self.taille_pas) * self.taille_pas
                    y_grille = round(y / self.taille_pas) * self.taille_pas

                    bordures_Robin.append((x_grille, y_grille))

                    # Calculer le vecteur normal (perpendiculaire à l'arête)
                    vecteur_arête = (p2[0] - p1[0], p2[1] - p1[1])
                    longueur_arête = math.hypot(vecteur_arête[0], vecteur_arête[1])
                    vecteur_normal = (
                        -vecteur_arête[1] / longueur_arête,
                        vecteur_arête[0] / longueur_arête,
                    )

                    # Translater le point le long du vecteur normal
                    x_translaté = x + self.taille_pas * vecteur_normal[0]
                    y_translaté = y + self.taille_pas * vecteur_normal[1]

                    # Vérifier si le point translaté est à l'intérieur du triangle
                    def signe(p1, p2, p3):
                        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
                            p1[1] - p3[1]
                        )

                    b1 = signe((x_translaté, y_translaté), PointA, PointB) < 0.0
                    b2 = signe((x_translaté, y_translaté), PointB, PointC) < 0.0
                    b3 = signe((x_translaté, y_translaté), PointC, PointA) < 0.0

                    est_dans = b1 == b2 == b3

                    # Inverser la direction du vecteur normal si le point est à l'intérieur du triangle
                    if est_dans:
                        x_translaté = x - self.taille_pas * vecteur_normal[0]
                        y_translaté = y - self.taille_pas * vecteur_normal[1]

                    # Aligner le point translaté sur la grille
                    x_grille_translaté = (
                        round(x_translaté / self.taille_pas) * self.taille_pas
                    )
                    y_grille_translaté = (
                        round(y_translaté / self.taille_pas) * self.taille_pas
                    )

                    # S'assurer que le point est dans les limites de l'espace discrétisé
                    if (
                        0 <= x_grille_translaté <= 300
                        and 0 <= y_grille_translaté <= 300
                    ):
                        bordures_Robin.append((x_grille_translaté, y_grille_translaté))

            print("Polygone ajouté")

        if self.type_forme == "cercle":
            x_centre, y_centre, rayon1, rayon2, taille_pas = (
                self.centre[0],
                self.centre[1],
                self.rayon1,
                self.rayon2,
                self.taille_pas,
            )
            points_cercle = []
            points_autre_cercle = []

            # Parcourir les points dans le rectangle englobant l'ellipse
            for y in range(
                int(y_centre - rayon2), int(y_centre + rayon2) + 1, self.taille_pas
            ):
                for x in range(
                    int(x_centre - rayon1), int(x_centre + rayon1) + 1, self.taille_pas
                ):
                    eq_ellipse = ((x - x_centre) ** 2 / rayon1**2) + (
                        (y - y_centre) ** 2 / rayon2**2
                    )
                    if eq_ellipse <= 1:
                        points_cercle.append((x, y))

            rayon1_plus = rayon1 + self.taille_pas
            rayon2_plus = rayon2 + self.taille_pas
            for y in range(
                int(y_centre - rayon2_plus), int(y_centre + rayon2_plus) + 1
            ):
                for x in range(
                    int(x_centre - rayon1_plus), int(x_centre + rayon1_plus) + 1
                ):
                    eq_ellipse_plus = ((x - x_centre) ** 2 / rayon1_plus**2) + (
                        (y - y_centre) ** 2 / rayon2_plus**2
                    )
                    if eq_ellipse_plus <= 1:
                        points_autre_cercle.append((x, y))

            for point in points_autre_cercle:
                if point not in points_cercle:
                    bordures_Robin.append(point)

        print("Condition de Robin ajoutée")

    def sauvegarder_forme(self):
        a = curseur_largeur.val
        b = curseur_hauteur.val
        if self.type_forme == "rectangle":
            self.ajouter_sous_rectangle(self.point1, self.point2)
            print("Rectangle sauvegardé")
            curseur_largeur.val, curseur_hauteur.val = 30, 30
            curseur_largeur.set_val(30)
            curseur_hauteur.set_val(30)
            self.forme_actuelle.set_width(a)
            self.forme_actuelle.set_height(b)

        if self.type_forme == "polygone":
            self.ajouter_sous_polygone(self.point4, self.point5, self.point6)
            x, y = self.forme_actuelle.xy[0]
            print("Triangle sauvegardé")
            curseur_largeur.set_val(30)
            curseur_hauteur.set_val(30)
            curseur_largeur.val, curseur_hauteur.val = 30, 30
            self.forme_actuelle.xy = [[x, y], [x - a, y], [x, y + b]]

        if self.type_forme == "cercle":
            self.ajouter_sous_cercle(self.centre, self.rayon1, self.rayon2)
            print("Ellipse sauvegardée")
            curseur_largeur.val, curseur_hauteur.val = 30, 30
            curseur_largeur.set_val(30)
            curseur_hauteur.set_val(30)
            self.forme_actuelle.set_width(a)
            self.forme_actuelle.set_height(b)


def definir_rectangle(evenement):
    editeur_formes.definir_type_forme("rectangle")


def definir_cercle(evenement):
    editeur_formes.definir_type_forme("cercle")


def definir_polygone(evenement):
    editeur_formes.definir_type_forme("polygone")


def callback_sauvegarde(evenement):
    condition_limite = messagebox.askyesno(
        "Robin", "Voulez-vous ajouter la condition limite de Robin à cette figure?"
    )

    if condition_limite:
        print("Condition limite de Robin ajoutée...")
        editeur_formes.ajouter_condition_Robin()
    else:
        print("Pas de condition limite de Robin.")

    editeur_formes.sauvegarder_forme()


def discretiser_espace(rectangle, sous_rects, sous_cercles, sous_polys, taille_pas):
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

    # Soustraire les points des sous-rectangles
    for sous_rect in sous_rects:
        x_min_sub, y_min_sub, x_max_sub, y_max_sub = sous_rect
        for x in range(int(x_min_sub), int(x_max_sub) + 1, taille_pas):
            for y in range(int(y_min_sub), int(y_max_sub) + 1, taille_pas):
                if (x, y) in points_figure:
                    points_figure.remove((x, y))

    # Soustraire les points des cercles
    for sous_cercle in sous_cercles:
        x_centre, y_centre, rayon1, rayon2 = sous_cercle
        for y in range(int(y_centre - rayon2), int(y_centre + rayon2) + 1):
            for x in range(int(x_centre - rayon1), int(x_centre + rayon1) + 1):
                eq_ellipse = ((x - x_centre) ** 2 / rayon1**2) + (
                    (y - y_centre) ** 2 / rayon2**2
                )
                if eq_ellipse <= 1:
                    if (x, y) in points_figure:
                        points_figure.remove((x, y))

    # Soustraire les points des polygones (triangles)
    for sous_tri in sous_polys:

        def point_dans_triangle(p, p1, p2, p3):
            """Détermine si un point est à l'intérieur d'un triangle en utilisant la méthode du signe."""

            def signe(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
                    p1[1] - p3[1]
                )

            d1 = signe(p, p1, p2)
            d2 = signe(p, p2, p3)
            d3 = signe(p, p3, p1)

            a_negatif = (d1 < 0) or (d2 < 0) or (d3 < 0)
            a_positif = (d1 > 0) or (d2 > 0) or (d3 > 0)

            return not (a_negatif and a_positif)

        pointA, pointB, pointC = sous_tri
        for point in list(points_figure):
            if point_dans_triangle(point, pointA, pointB, pointC):
                points_figure.remove(point)

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


def assigner_conditions_limites(rectangle, sous_rects, taille_pas):
    """Assigne les conditions limites aux points de la discrétisation."""
    conditions_limites = {}

    # Conditions sur les bords du rectangle principal
    for point in [(0, y) for y in range(rectangle[1], rectangle[3] + 1, taille_pas)]:
        conditions_limites[point] = 1  # Condition de type 1

    for point in [
        (rectangle[2], y) for y in range(rectangle[1], rectangle[3] + 1, taille_pas)
    ]:
        conditions_limites[point] = 2  # Condition de Robin

    # Ajouter les bordures de Robin définies par l'utilisateur
    for point in bordures_Robin:
        conditions_limites[point] = 2

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

    quiver = plt.quiver(
        x_echantillones,
        y_echantillones,
        flux_x,
        flux_y,
        norme_flux,
        angles="xy",
        scale_units="xy",
        scale=0.18,
        cmap="viridis",
    )
    plt.colorbar(quiver, label="Module du flux (-D.grad(C))")
    plt.title("Champ de flux (-D.grad(C)) en mmol/(m²s)")
    plt.xlabel("x (10^-2 mm)")
    plt.ylabel("y (10^-2 mm)")
    plt.grid(True)
    plt.show()


def tracer_points_conditions(coord_x, coord_y, conditions):
    """Trace les points avec leurs conditions limites."""
    couleurs = [conditions.get((x, y), 0) for x, y in zip(coord_x, coord_y)]
    plt.scatter(coord_x, coord_y, c=couleurs, cmap="plasma")
    plt.title("Discrétisation de l'espace avec les conditions aux limites")
    plt.xlabel("Position x (en 10^-2 mm)")
    plt.ylabel("Position y (en 10^-2 mm)")
    plt.grid(True)
    plt.show()


def selection_rectangle(eclic, erelease):
    global rectangle_actuel
    rectangle_actuel = [eclic.xdata, eclic.ydata, erelease.xdata, erelease.ydata]


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
    plt.grid(True)
    plt.show()


# test_convergence([10, 5, 2, 1])


def main():
    rectangle_principal = (0, 0, 300, 300)  # Unités en 10^-2 mm
    taille_pas = 5
    pas_discretisation = taille_pas * 0.01
    figure, axe = plt.subplots()
    global editeur_formes
    editeur_formes = EditeurFormes(axe, rectangle_principal, taille_pas)
    figure.subplots_adjust(left=0.05, right=0.65, top=0.95, bottom=0.2)

    # Créer des boutons pour chaque type de forme, placés à droite
    axe_bouton_rect = plt.axes([0.7, 0.75, 0.25, 0.075])  # Bouton Rectangle
    axe_bouton_cercle = plt.axes([0.7, 0.65, 0.25, 0.075])  # Bouton Cercle
    axe_bouton_polygone = plt.axes([0.7, 0.55, 0.25, 0.075])  # Bouton Polygone
    axe_bouton_sauvegarder = plt.axes([0.7, 0.05, 0.1, 0.075])  # Bouton Sauvegarder
    btn_sauvegarder = Button(axe_bouton_sauvegarder, "Sauvegarder")
    btn_rectangle = Button(axe_bouton_rect, "Rectangle")
    btn_cercle = Button(axe_bouton_cercle, "Ellipse")
    btn_polygone = Button(axe_bouton_polygone, "Polygone")

    # Attacher les fonctions de callback aux boutons
    btn_rectangle.on_clicked(definir_rectangle)
    btn_cercle.on_clicked(definir_cercle)
    btn_polygone.on_clicked(definir_polygone)
    btn_sauvegarder.on_clicked(callback_sauvegarde)

    # Créer des curseurs pour la largeur et la hauteur, placés en bas
    axe_curseur_largeur = plt.axes(
        [0.1, 0.05, 0.55, 0.03], facecolor="lightgoldenrodyellow"
    )
    axe_curseur_hauteur = plt.axes(
        [0.1, 0.1, 0.55, 0.03], facecolor="lightgoldenrodyellow"
    )

    global curseur_largeur, curseur_hauteur
    curseur_largeur = Slider(
        axe_curseur_largeur, "Largeur/Rayon", 5.0, 200.0, valinit=30
    )
    curseur_hauteur = Slider(axe_curseur_hauteur, "Hauteur", 5.0, 200.0, valinit=30)

    # Connecter les curseurs à l'éditeur de formes pour mettre à jour la taille des formes
    curseur_largeur.on_changed(editeur_formes.mettre_a_jour_taille_forme)
    curseur_hauteur.on_changed(editeur_formes.mettre_a_jour_taille_forme)

    plt.show()

    # Récupérer les sous-formes ajoutées
    sous_rectangles = editeur_formes.sous_rectangles
    sous_polygones = editeur_formes.sous_polygones
    sous_cercles = editeur_formes.sous_cercles

    print("Sous-rectangles ajoutés:")
    for sous_rect in sous_rectangles:
        print(sous_rect)
    print("Sous-triangles ajoutés:")
    for sous_tri in sous_polygones:
        print(sous_tri)
    print("Sous-ellipses ajoutées:")
    for sous_cercle in sous_cercles:
        print(sous_cercle)

    # Discrétiser l'espace
    points_discrets, voisins, coefficients = discretiser_espace(
        rectangle_principal, sous_rectangles, sous_cercles, sous_polygones, taille_pas
    )
    conditions_limites = assigner_conditions_limites(
        rectangle_principal, sous_rectangles, taille_pas
    )

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

    tracer_flux(
        points_discrets,
        vecteur_resultat,
        voisins,
        taille_pas,
        diffusivite,
        intervalle=1,
    )

    # Définir les valeurs fixes pour les graphiques de débit

    debit_total_entree = 0
    debit_total_sortie = 0
    nombre_points = len(points_discrets)

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

    print(
        "Valeur du débit correspondant à la figure choisie en ml/s (entrée et sortie)"
    )
    print("debit entrant ", debit_total_entree)
    print("debit sortant ", debit_total_sortie)


if __name__ == "__main__":
    main()
