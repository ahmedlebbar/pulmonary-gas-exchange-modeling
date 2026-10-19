import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from scipy.sparse import lil_matrix, csc_matrix  
from scipy.sparse.linalg import spsolve  

# Constants and parameters
M = 200
N = 200
R = M * N
L = 0.045
l = 0.00655
P1 = 3.8e-4
P2 = 0
Dx = L / (M - 1)
Dy = l / (N - 1)
e = 0.0000185

# Centres et rayons des sphères définissant la région Deff
center1_x, center1_y = 150, 0
center2_x, center2_y = 80, N
center3_x, center3_y = 110, N
radius1 = 40  # Rayon de la première sphère
radius2 = 20  # Rayon de la deuxième sphère
radius3 = 30  # Rayon de la troisième sphère

# Définition de Deff : ensemble des points à l'intérieur des sphères définies
Deff = [
    (i, j)
    for i in range(1, M + 1)  # Parcours des indices i
    for j in range(1, N + 1)  # Parcours des indices j
    if (
        (((i - center1_x) ** 2 + ((j - center1_y) * 0.6) ** 2 <= radius1 ** 2) or
         ((i - center2_x) ** 2 + ((j - center2_y) * 0.6) ** 2 <= radius2 ** 2) or
         ((i - center3_x) ** 2 + ((j - center3_y) * 0.6) ** 2 <= radius3 ** 2))
    )  # Vérifie si le point est dans l'une des sphères
]

# Définition des points de frontière et autres ensembles
I = {(1, 1), (1, N), (M, 1), (M, N)}  # Coins du domaine
A = {(i, N) for i in range(2, M) if (i, N) not in Deff}  # Bord supérieur excluant Deff
B = {(i, 1) for i in range(2, M) if (i, 1) not in Deff}  # Bord inférieur excluant Deff
C = {(1, j) for j in range(2, N)}  # Bord gauche excluant les coins
D = {(M, j) for j in range(2, N)}  # Bord droit excluant les coins
E2 = {(i, j) for i in range(2, M) for j in range(2, N)}  # Ensemble interne

# Identification des contours de Deff pour appliquer les conditions de frontière
Deff_contour_up = []
Deff_contour_down = []
Deff_contour_right = []
Deff_contour_left = []

for (i, j) in Deff:
    if ((i + 1, j) not in Deff and (i + 1, j) not in D):
        Deff_contour_right.append((i + 1, j))
    if ((i - 1, j) not in Deff and (i - 1, j) not in C):
        Deff_contour_left.append((i - 1, j))
    if ((i, j + 1) not in Deff and (i, j + 1) not in A and (i, j + 1) not in B):
        Deff_contour_up.append((i, j + 1))
    if ((i, j - 1) not in Deff and (i, j - 1) not in A and (i, j - 1) not in B):
        Deff_contour_down.append((i, j - 1))

# Combinaison des contours de Deff
Deff_contour = Deff_contour_left + Deff_contour_right + Deff_contour_down + Deff_contour_up

# Définition de l'ensemble E excluant Deff et ses contours
E = {(i, j) for i in range(2, M) for j in range(2, N) if (i, j) not in Deff and (i, j) not in Deff_contour}

def bij(i, j):
    """Convertit les indices (i, j) en un indice unique."""
    return (j - 1) + N * (i - 1)

def bij_inverse(idx):
    """Convertit un indice unique en indices (i, j)."""
    j = (idx % N) + 1  # Indice de colonne
    i = (idx // N) + 1  # Indice de ligne
    return i, j

# Initialisation de la matrice creuse Z et du vecteur Y
Z = lil_matrix((3 * M * N, 3 * M * N))
Y = np.zeros(3 * M * N)

# Assemblage des équations du système linéaire
for (i, j) in E:
    idx = bij(i, j)
    # Première équation : Relation pour Ux
    Z[idx, bij(i + 1, j) + 2 * R] = -(1) / (Dx * 2)
    Z[idx, bij(i - 1, j) + 2 * R] = (1) / (Dx * 2)
    Z[idx, bij(i + 1, j)] = e / (Dx ** 2)
    Z[idx, bij(i - 1, j)] = e / (Dx ** 2)
    Z[idx, bij(i, j + 1)] = e / (Dy ** 2)
    Z[idx, bij(i, j - 1)] = e / (Dy ** 2)
    Z[idx, idx] = -2 * e * ((1 / (Dx ** 2) + 1 / (Dy ** 2)))
    
    # Deuxième équation : Relation pour Uy
    Z[idx + R, bij(i, j + 1) + 2 * R] = -(1) / (Dy * 2)
    Z[idx + R, bij(i, j - 1) + 2 * R] = (1) / (Dy * 2)
    Z[idx + R, bij(i + 1, j) + R] = e / (Dx ** 2)
    Z[idx + R, bij(i - 1, j) + R] = e / (Dx ** 2)
    Z[idx + R, bij(i, j + 1) + R] = e / (Dy ** 2)
    Z[idx + R, bij(i, j - 1) + R] = e / (Dy ** 2)
    Z[idx + R, idx + R] = -2 * e * ((1 / (Dx ** 2) + 1 / (Dy ** 2)))
    
    # Troisième équation : Relation pour la pression P
    Z[idx + 2 * R, bij(i + 1, j)] = 1 / (Dx * 2)
    Z[idx + 2 * R, bij(i - 1, j)] = -1 / (Dx * 2)
    Z[idx + 2 * R, bij(i, j + 1) + R] = 1 / (Dy * 2)
    Z[idx + 2 * R, bij(i, j - 1) + R] = -1 / (Dy * 2)

# Application des conditions aux coins du domaine
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
    Z[idx + 2 * R, idx + 2 * R] = -e * (1 / (Dx ** 2)) - (1 / (Dy ** 2))
    Z[idx + 2 * R, idx + 2 * R + 1] = 1 / (Dy ** 2)
    Z[idx + 2 * R, idx + 2 * R + N] = 1 / (Dx ** 2)
    Z[idx + 2 * R, idx + 2 * R - N] = 1 / (Dx ** 2)

# Application des conditions sur le bord gauche
for i, j in C:
    idx = bij(i, j)
    Z[idx, bij(i + 1, j) + 2 * R] = -1 / Dx
    Z[idx, bij(i, j) + 2 * R] = 1 / Dx
    Z[idx, bij(i + 1, j)] = e / (Dx ** 2)
    Z[idx, bij(M - 1, j)] = e / (Dx ** 2)
    Z[idx, bij(i, j + 1)] = e / (Dy ** 2)
    Z[idx, bij(i, j - 1)] = e / (Dy ** 2)
    Z[idx, idx] = -e * ((2 / (Dx ** 2) + 2 / (Dy ** 2)))
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1
    Y[idx + 2 * R] = P1  # Condition de pression

# Application des conditions sur le bord droit
for i, j in D:
    idx = bij(i, j)
    Z[idx, bij(i, j) + 2 * R] = -1 / Dx
    Z[idx, bij(i - 1, j) + 2 * R] = 1 / Dx
    Z[idx, bij(2, j)] = e / (Dx ** 2)
    Z[idx, bij(i - 1, j)] = e / (Dx ** 2)
    Z[idx, bij(i, j + 1)] = e / (Dy ** 2)
    Z[idx, bij(i, j - 1)] = e / (Dy ** 2)
    Z[idx, idx] = -e * ((2 / (Dx ** 2) + 2 / (Dy ** 2)))
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1
    Y[idx + 2 * R] = P2  # Condition de pression

# Définition des conditions aux coins supplémentaires
Y[bij(1, 1) + 2 * R] = P1
Y[bij(1, N) + 2 * R] = P1
Y[bij(M, N) + 2 * R] = P2
Y[bij(M, 1) + 2 * R] = P2

# Application des conditions de Deff
for (i, j) in Deff:
    idx = bij(i, j)
    Z[idx, idx] = 1
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1

# Application des conditions sur le contour de Deff
for (i, j) in Deff_contour:
    idx = bij(i, j)
    Z[idx + 2 * R, bij(i + 1, j)] = 1 / (Dx * 2)
    Z[idx + 2 * R, bij(i - 1, j)] = -1 / (Dx * 2)
    Z[idx + 2 * R, bij(i, j + 1) + R] = 1 / (Dy * 2)
    Z[idx + 2 * R, bij(i, j - 1) + R] = -1 / (Dy * 2)
    Z[idx + R, idx + R] = 1
    Z[idx, idx] = 1

# Définition des conditions aux coins restants
Y[2 * R] = P1
Y[2 * R + N - 1] = P1
Y[3 * R - 1] = P2
Y[3 * R - N] = P2

def detect_zero_columns(matrix):
    """Détecte les colonnes contenant uniquement des zéros dans la matrice."""
    zero_columns = []
    # Conversion en format colonne creuse pour une accès efficace
    if not isinstance(matrix, csc_matrix):
        matrix = csc_matrix(matrix)
    
    # Parcours des colonnes spécifiques pour détecter les colonnes nulles
    for j in range(2 * R - 1, 3 * R):
        if matrix.getcol(j).nnz == 0:
            zero_columns.append(j)
    
    return zero_columns

# Identification des colonnes nulles dans la matrice Z
zero_columns = detect_zero_columns(Z)

# Correction des colonnes nulles détectées
for j in zero_columns:
    for i in range(2 * R, 3 * R):
        Z[j, i] = 0
        k, f = bij_inverse(j)
    if bij_inverse(j) in A:
        Z[j, j] = -e * (1 / (Dx ** 2)) - (1 / (Dy ** 2))
        Z[j, j - 1] = 1 / (Dy ** 2)
        Z[j, j + N] = 1 / (Dx ** 2)
        Z[j, j - N] = 1 / (Dx ** 2)
    elif bij_inverse(j) in B:
        Z[j, j] = -e * (1 / (Dx ** 2)) - (1 / (Dy ** 2))
        Z[j, j + 1] = 1 / (Dy ** 2)
        Z[j, j + N] = 1 / (Dx ** 2)
        Z[j, j - N] = 1 / (Dx ** 2)
    else:
        Z[j, j] = -e * (1 / (Dx ** 2)) - (1 / (Dy ** 2))
        Z[j, j + 1] = 1 / (Dy ** 2)
        Z[j, j - 1] = 1 / (Dy ** 2)
        Z[j, j + N] = 1 / (Dx ** 2)
        Z[j, j - N] = 1 / (Dx ** 2)

# Résolution du système linéaire Z * X = Y
Z = csc_matrix(Z)  # Conversion en format colonne creuse pour l'algorithme de résolution
X = spsolve(Z, Y)  # Résolution du système

# Extraction des composantes Ux, Uy et P à partir de la solution
Ux = X[:M * N].reshape((M, N))
Uy = X[M * N:2 * M * N].reshape((M, N))
P = X[2 * M * N:].reshape((M, N))
P_smoothed = P.copy()

# Visualisation des résultats : Ux, Uy et Pression P en 2D
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Affichage de Ux
img1 = ax[0].imshow(Ux, cmap='viridis', aspect='auto')
fig.colorbar(img1, ax=ax[0])
ax[0].set_title('Variation de Ux')
ax[0].set_xlabel('Index Y')
ax[0].set_ylabel('Index X')

# Affichage de Uy
img2 = ax[1].imshow(Uy, cmap='viridis', aspect='auto')
fig.colorbar(img2, ax=ax[1])
ax[1].set_title('Variation de Uy')
ax[1].set_xlabel('Index Y')
ax[1].set_ylabel('Index X')

# Affichage de la pression P
img3 = ax[2].imshow(P_smoothed, cmap='viridis', aspect='auto')
fig.colorbar(img3, ax=ax[2])
ax[2].set_title('Pression P')
ax[2].set_xlabel('Index Y')
ax[2].set_ylabel('Index X')

# Création de la grille pour les axes X et Y
X_grid, Y_grid = np.meshgrid(np.arange(N), np.arange(M))

# Visualisation des résultats en 3D
fig = plt.figure(figsize=(18, 6))

# Affichage 3D de Ux
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Ux, cmap='viridis')
ax1.set_title('Variation de Ux (3D)')
ax1.set_xlabel('Index Y')
ax1.set_ylabel('Index X')
ax1.set_zlabel('Ux')

# Affichage 3D de Uy
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Uy, cmap='viridis')
ax2.set_title('Variation de Uy (3D)')
ax2.set_xlabel('Index Y')
ax2.set_ylabel('Index X')
ax2.set_zlabel('Uy')

# Affichage 3D de la pression P
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_grid, Y_grid, P, cmap='viridis')
ax3.set_title('Pression P (3D)')
ax3.set_xlabel('Index Y')
ax3.set_ylabel('Index X')
ax3.set_zlabel('Pression P')

# Création de la grille pour la visualisation en 2D avec quiver
X_grid, Y_grid = np.meshgrid(np.arange(N), np.arange(M))

# Visualisation de la pression avec les vecteurs de vitesse
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(Y_grid, X_grid, P, cmap='viridis', alpha=0.7)  # Contour de la pression en arrière-plan
fig.colorbar(contour, ax=ax, label='Pression P')

# Définition du pas de sous-échantillonnage pour les vecteurs de vitesse
sampling_step = 5  # Contrôle la densité des flèches

# Sous-échantillonnage des données de vitesse
Ux_subsampled = Ux[::sampling_step, ::sampling_step]
Uy_subsampled = Uy[::sampling_step, ::sampling_step]

# Sous-échantillonnage de la grille
X_grid_subsampled = X_grid[::sampling_step, ::sampling_step]
Y_grid_subsampled = Y_grid[::sampling_step, ::sampling_step]

# Tracé des vecteurs de vitesse sur le contour de pression
ax.quiver(Y_grid_subsampled, X_grid_subsampled, Ux_subsampled, Uy_subsampled, scale_units='xy', angles='xy', color='blue')
ax.set_title('Quiver plot: Vecteurs de vitesse (Ux, Uy) sous-échantillonnés')
ax.set_xlabel('Index Y')
ax.set_ylabel('Index X')

plt.show()
