import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve 

# Constantes et paramètres du modèle
M = 100  # Nombre de points en x
N = 100  # Nombre de points en y
R = M * N  # Taille totale des points
L = 0.049  # Longueur dans la direction x (m)
l = 0.0067  # Longueur dans la direction y (m)
P1 = 3.8e-4  # Pression à une frontière
P2 = 0  # Pression à une autre frontière
dt = 0.01  # Pas de temps (s)
temps_indice = 200  # Nombre d'itérations temporelles
Dx = L / (M - 1)  # Pas spatial en x
Dy = l / (N - 1)  # Pas spatial en y
e = 0.0000185  # viscosité dynamique
matrice_debit = []  # Liste pour stocker l'évolution du débit
f = 1  # Indice temporel pour le débit

# Configuration du répertoire de sortie

# Le GIF est enregistré dans le dossier results/ du projet
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
os.makedirs(output_dir, exist_ok=True)


# Fonction retournant une valeur pour les calculs dépendant du temps
def fonction_temps(index):
    return 0  # Fonction exemple, à modifier si nécessaire

# Définition des points de frontière et des ensembles dans un domaine 2D
I = {(1, 1), (1, N), (M, 1), (M, N)}  # Quatre coins du domaine
A = {(i, N) for i in range(2, M)}  # Bord supérieur excluant les coins
B = {(i, 1) for i in range(2, M)}  # Bord inférieur excluant les coins
C = {(1, j) for j in range(2, N)}  # Bord gauche excluant les coins
D = {(M, j) for j in range(2, N)}  # Bord droit excluant les coins
E = {(i, j) for i in range(2, M) for j in range(2, N)}  # Points intérieurs

# Conversion des indices 2D en indices 1D
def bij(i, j):
    return (j - 1) + N * (i - 1)

# Création de la matrice Z et des vecteurs X et Y
Z = lil_matrix((3 * M * N, 3 * M * N))
X = np.zeros(3 * M * N)
Y = np.zeros(3 * M * N)

# Création d'un maillage pour le tracé de flèches (quiver)
X_grid, Y_grid = np.meshgrid(np.linspace(0, L, M), np.linspace(0, l, N))

# Préparation de l'animation avec des axes séparés
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Création d'une carte des couleurs pour la pression (ax2)
pressure_plot = ax2.pcolor(X_grid, Y_grid, np.zeros((N, M)), shading='auto', cmap='viridis', alpha=0.5)
ax2.set_title('Champ de pression')
plt.colorbar(pressure_plot, ax=ax2, label='Pression')

# Tracé des vecteurs de vitesse (ax1)
quiver = ax1.quiver(X_grid, Y_grid, np.zeros_like(X_grid), np.zeros_like(Y_grid), scale_units='xy', angles='xy')
ax1.set_title('Champ de vitesse')

# Redéfinition du quiver avec des paramètres ajustés
quiver = ax1.quiver(
    X_grid, Y_grid, np.zeros_like(X_grid), np.zeros_like(Y_grid),
    scale=1, scale_units='xy', angles='xy'
)
ax1.set_title('Champ de vitesse')

# Remplissage de la matrice Z pour les points intérieurs
for (i, j) in E:
    idx = bij(i, j)
    # Première équation de discretisation
    Z[idx, bij(i, j) + 2 * R] = -(1) / (Dx)
    Z[idx, bij(i - 1, j) + 2 * R] = (1) / (Dx)
    Z[idx, bij(i + 1, j)] = e / (Dx**2)
    Z[idx, bij(i - 1, j)] = e / (Dx**2)
    Z[idx, bij(i, j + 1)] = e / (Dy**2)
    Z[idx, bij(i, j - 1)] = e / (Dy**2)
    Z[idx, idx] = -2 * e * ((1 / (Dx**2) + 1 / (Dy**2))) - 1 / dt
    # Deuxième équation de discretisation
    Z[idx + R, bij(i, j + 1) + 2 * R] = -(1) / (Dy * 2)
    Z[idx + R, bij(i, j - 1) + 2 * R] = (1) / (Dy * 2)
    Z[idx + R, bij(i + 1, j) + R] = e / (Dx**2)
    Z[idx + R, bij(i - 1, j) + R] = e / (Dx**2)
    Z[idx + R, bij(i, j + 1) + R] = e / (Dy**2)
    Z[idx + R, bij(i, j - 1) + R] = e / (Dy**2)
    Z[idx + R, idx + R] = -2 * e * ((1 / (Dx**2) + 1 / (Dy**2))) - 1 / dt

    # Troisième équation de discretisation (relation de vitesse)
    Z[idx + 2 * R, bij(i + 1, j)] = 1 / (Dx * 2)
    Z[idx + 2 * R, bij(i - 1, j)] = -1 / (Dx * 2)
    Z[idx + 2 * R, bij(i, j + 1) + R] = 1 / (Dy * 2)
    Z[idx + 2 * R, bij(i, j - 1) + R] = -1 / (Dy * 2)

# Application des conditions aux coins
for (i, j) in I:
    idx = bij(i, j)
    Z[idx, idx] = 1
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1

# Application des conditions sur le bord supérieur
for (i, j) in A:
    idx = bij(i, j)
    Z[idx, idx] = 1
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, bij(i + 1, j)] = 1 / (Dx * 2)
    Z[idx + 2 * R, bij(i - 1, j)] = -1 / (Dx * 2)
    Z[idx + 2 * R, idx + R] = 1 / Dy
    Z[idx + 2 * R, bij(i, j - 1) + R] = -1 / Dy

# Application des conditions sur le bord inférieur
for (i, j) in B:
    idx = bij(i, j)
    Z[idx, idx] = 1
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, bij(i + 1, j)] = 1 / (Dx * 2)
    Z[idx + 2 * R, bij(i - 1, j)] = -1 / (Dx * 2)
    Z[idx + 2 * R, bij(i, j + 1) + R] = 1 / Dy
    Z[idx + 2 * R, idx + R] = -1 / Dy

# Application des conditions sur le bord gauche
for i, j in C:
    idx = bij(i, j)
    Z[idx, bij(i + 1, j) + 2 * R] = -1 / Dx
    Z[idx, bij(i, j) + 2 * R] = 1 / Dx
    Z[idx, bij(i + 1, j)] = e / (Dx ** 2)
    Z[idx, bij(M - 1, j)] = e / (Dx ** 2)
    Z[idx, bij(i, j + 1)] = e / (Dy ** 2)
    Z[idx, bij(i, j - 1)] = e / (Dy ** 2)
    Z[idx, idx] = -e * ((2 / (Dx ** 2) + 2 / (Dy ** 2))) - 1 / dt
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1
    Y[idx + 2 * R] = P1  # Condition de pression sur le bord gauche

# Application des conditions sur le bord droit
for i, j in D:
    idx = bij(i, j)
    Z[idx, bij(i, j) + 2 * R] = -1 / Dx
    Z[idx, bij(i - 1, j) + 2 * R] = 1 / Dx
    Z[idx, bij(2, j)] = e / (Dx ** 2)
    Z[idx, bij(i - 1, j)] = e / (Dx ** 2)
    Z[idx, bij(i, j + 1)] = e / (Dy ** 2)
    Z[idx, bij(i, j - 1)] = e / (Dy ** 2)
    Z[idx, idx] = -e * ((2 / (Dx ** 2) + 2 / (Dy ** 2))) - 1 / dt
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1
    Y[idx + 2 * R] = P2  # Condition de pression sur le bord droit

# Conditions de pression aux coins
Y[bij(1, 1) + 2 * R] = P1
Y[bij(1, N) + 2 * R] = P1
Y[bij(M, N) + 2 * R] = P2
Y[bij(M, 1) + 2 * R] = P2

# Conversion de la matrice Z en format CSC pour l'efficacité des calculs
Z = csc_matrix(Z)

def update(k):
    global X
    global f
    if k == 0:
        X = np.zeros(3 * M * N)  # Initialisation du vecteur solution
    else:
        Y.fill(0)  # Réinitialisation du vecteur Y
        for (i, j) in E:
            idx = bij(i, j)
            Y[idx] = -X[idx] / dt 
            Y[idx + R] = -X[idx + R] / dt  

        # Application des conditions aux frontières pour chaque itération
        Y[bij(1, 1)] = fonction_temps(k)
        Y[bij(1, N)] = fonction_temps(k)
        Y[bij(M, N)] = fonction_temps(k)
        Y[bij(M, 1)] = fonction_temps(k)

        # Conditions sur le bord supérieur
        for (i, j) in A:
            idx = bij(i, j)
            Y[idx] = fonction_temps(k)
            Y[idx + R] = 0

        # Conditions sur le bord inférieur
        for (i, j) in B:
            idx = bij(i, j)
            Y[idx] = fonction_temps(k)
            Y[idx + R] = 0

        # Conditions sur le bord gauche
        for (i, j) in C:
            idx = bij(i, j)
            Y[idx] = -X[idx] / dt
            Y[idx + 2 * R] = P1

        # Conditions sur le bord droit
        for (i, j) in D:
            idx = bij(i, j)
            Y[idx] = -X[idx] / dt
            Y[idx + 2 * R] = P2

        # Conditions de pression aux coins
        Y[bij(1, 1) + 2 * R] = P1
        Y[bij(1, N) + 2 * R] = P1
        Y[bij(M, N) + 2 * R] = P2
        Y[bij(M, 1) + 2 * R] = P2

        # Résolution du système linéaire pour obtenir le nouveau vecteur solution
        X = spsolve(Z, Y)
        print(k * dt)  # Affichage du temps courant

    # Mise à jour des graphiques
    P = X[2 * R:3 * R].reshape(N, M)  # Extraction des valeurs de pression
    pressure_plot.set_array(np.transpose(P))  # Mise à jour de la carte de pression
    pressure_plot.autoscale()  # Ajustement de l'échelle des couleurs

    # Extraction et mise à jour des vecteurs de vitesse
    Ux = X[:R].reshape(N, M)
    Uy = X[R:2 * R].reshape(N, M)
    quiver.set_UVC(np.transpose(Ux), np.transpose(Uy))  # Mise à jour des flèches de vitesse

    # Calcul et stockage du débit si dans les limites temporelles
    debit = 0
    if f <= temps_indice:
        for j in range(N):
            debit += Ux[0, j] * Dy  
        print(str(debit))
        print(k)
        matrice_debit.append(debit)
        f += 1

    return pressure_plot, quiver

# Création de l'animation et sauvegarde au format GIF
anim = FuncAnimation(fig, update, frames=temps_indice, interval=100, blit=True)
anim.save(os.path.join(output_dir, "stokes_transient.gif"), writer="pillow", fps=60)

plt.show()

# Tracé de l'évolution du débit en fonction du temps
S = [i * dt for i in range(0, f - 1)]
plt.plot(S, matrice_debit, '--b', label='Débit numérique')
plt.axhline(y=1.04203e-5, color='red', linestyle='--', label='Débit stationnaire')
plt.axhline(y=0.93503e-5, color='green', linestyle='--', label='Débit moyen final')
plt.xlabel('Temps (s)')
plt.ylabel('Débit')
plt.title("Évolution du débit en fonction du temps")
plt.grid(True)
plt.legend()
plt.show()
