import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# Constants and parameters
M = 150
N = 150
R = M * N
L = 0.049
l = 0.0067
P1 = 10+3.8e-4
P2 = 10
Dx = L / (M - 1)
Dy = l / (N - 1)
e = 0.0000185
debit = []


# Boundary points and sets in 2D
I = {(1, 1), (1, N), (M, 1), (M, N)}  # Four corners
A = {(i,N) for i in range(2, M)}  # Top edge excluding corners
B = {(i, 1) for i in range(2, M)}
C = {(1, j) for j in range(2, N)}  # Left edge excluding corners
D = {(M, j) for j in range(2, N)}  # Right edge excluding corners
E = {(i, j) for i in range(2, M) for j in range(2, N)}  # Interior points

# Convert 2D indices to 1D
def bij(i, j):
    return (j - 1) + N * (i - 1)

# Create the matrix Z and vector Y
Z = lil_matrix((3 * M * N, 3 * M * N))
Y = np.zeros(3 * M * N)

# Populate the matrix Z
for (i,j) in E:
    idx = bij(i,j)
    # First equation:
    Z[idx,bij(i,j)+2*R] = -(1)/(Dx)
    Z[idx,bij(i-1,j)+2*R] =  (1)/(Dx)
    Z[idx,bij(i+1,j)] = e/(Dx**2)
    Z[idx,bij(i-1,j)] = e/(Dx**2)
    Z[idx,bij(i,j+1)] = e/(Dy**2)
    Z[idx,bij(i,j-1)] = e/(Dy**2)
    Z[idx,idx] = -2*e*((1/(Dx**2)+1/(Dy**2)))
    # Second equation:
    Z[idx+R,bij(i,j+1)+2*R] = -(1)/(Dy*2)
    Z[idx+R,bij(i,j-1)+2*R] =  (1)/(Dy*2)
    Z[idx+R,bij(i+1,j)+R] = e/(Dx**2)
    Z[idx+R,bij(i-1,j)+R] = e/(Dx**2)
    Z[idx+R,bij(i,j+1)+R] = e/(Dy**2)
    Z[idx+R,bij(i,j-1)+R] = e/(Dy**2)
    Z[idx+R,idx+R] = -2*e*((1/(Dx**2)+1/(Dy**2)))

    # Third equation:
    Z[idx+2*R,bij(i+1,j)] = 1/(Dx*2)
    Z[idx+2*R,bij(i-1,j)] = -1/(Dx*2)
    Z[idx+2*R,bij(i,j+1)+R] = 1/(Dy*2)
    Z[idx+2*R,bij(i,j-1)+R] = -1/(Dy*2)

for (i,j) in I:
    idx = bij(i,j)
    Z[idx,idx]=1
    Z[idx+R,idx+R]=1
    Z[idx+2*R,idx+2*R]=1
    
# Top edge
for (i,j) in A:
    idx = bij(i,j)
    Z[idx,idx]=1
    Z[idx+R,idx+R]=1
    Z[idx+2*R,bij(i+1,j)] = 1/(Dx*2)
    Z[idx+2*R,bij(i-1,j)] = -1/(Dx*2)
    Z[idx+2*R,idx+R] = 1/(Dy)
    Z[idx+2*R,bij(i,j-1)+R] = -1/(Dy)

# Bottom edge
for (i,j) in B:
    idx = bij(i,j)
    Z[idx,idx]=1
    Z[idx+R,idx+R]=1
    Z[idx+2*R,bij(i+1,j)] = 1/(Dx*2)
    Z[idx+2*R,bij(i-1,j)] = -1/(Dx*2)
    Z[idx+2*R,bij(i,j+1)+R] = 1/(Dy)
    Z[idx+2*R,idx+R] = -1/(Dy)

for i, j in C:
    idx = bij(i, j)
    Z[idx, bij(i + 1, j) + 2 * R] = -1 / Dx
    Z[idx, bij(i, j) + 2 * R] = 1 / Dx
    Z[idx, bij(i + 1, j)] = e / (Dx ** 2)
    Z[idx, bij(M-1, j)] = e / (Dx ** 2)
    Z[idx, bij(i, j + 1)] = e / (Dy ** 2)
    Z[idx, bij(i, j - 1)] = e / (Dy ** 2)
    Z[idx, idx] = -e * ((2 / (Dx ** 2) + 2 / (Dy ** 2)))
    Z[idx + R, idx + R] = 1
    Z[idx + 2 * R, idx + 2 * R] = 1
    Y[idx + 2 * R] = P1

# Right edge
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
    Y[idx + 2 * R] = P2


Y[bij(1, 1) + 2 * R] = P1
Y[bij(1, N) + 2 * R] = P1
Y[bij(M, N) + 2 * R] = P2
Y[bij(M, 1) + 2 * R] = P2


# Output the results
# Solve the system Z * X = Y
Z = csc_matrix(Z)

# Solve the system Z * X = Y
X = spsolve(Z, Y)

# Extract Ux, Uy, and P from the solution X
Ux = X[:M * N].reshape((M, N))
Uy = X[M * N:2 * M * N].reshape((M, N))
P =  X[2 * M * N:].reshape((M, N))

P_smoothed = P.copy()

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(15, 6))

# Ux plot
img1 = ax[0].imshow(Ux, cmap='viridis', aspect='auto')
fig.colorbar(img1, ax=ax[0])
ax[0].set_title('Variation de Ux m/s')
ax[0].set_xlabel('Index Y')
ax[0].set_ylabel('Index X')

# Uy plot
img2 = ax[1].imshow(Uy, cmap='viridis', aspect='auto')
fig.colorbar(img2, ax=ax[1])
ax[1].set_title('Variation de Uy m/s')
ax[1].set_xlabel('Index Y')
ax[1].set_ylabel('Index X')

# Pressure plot
img3 = ax[2].imshow(P_smoothed, cmap='viridis', aspect='auto')
fig.colorbar(img3, ax=ax[2])
ax[2].set_title('Pression P en pascal')
ax[2].set_xlabel('Index Y')
ax[2].set_ylabel('Index X')



# Create the grid for X and Y values
X_grid, Y_grid = np.meshgrid(np.arange(N), np.arange(M))

# Plot the results in 3D
fig = plt.figure(figsize=(14, 6))

# 3D Ux plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Ux, cmap='viridis')
ax1.set_title('Variation de Ux (3D) en m/s')
ax1.set_xlabel('Index Y')
ax1.set_ylabel('Index X')
ax1.set_zlabel('Ux')

# 3D Uy plot
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Uy, cmap='viridis')
ax2.set_title('Variation de Uy (3D) en m/s')
ax2.set_xlabel('Index Y')
ax2.set_ylabel('Index X')
ax2.set_zlabel('Uy')

# 3D Pressure plot
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_grid, Y_grid, P, cmap='viridis')
ax3.set_title('Pression P (3D) en pascal')
ax3.set_xlabel('Index Y')
ax3.set_ylabel('Index X')
ax3.set_zlabel('Pression P')
# Create the grid for X and Y values
X_grid, Y_grid = np.meshgrid(np.arange(N), np.arange(M))

# Plot the pressure variation as a background using contourf or imshow
fig, ax = plt.subplots(figsize=(7, 6))
contour = ax.contourf(Y_grid, X_grid, P, cmap='viridis', alpha=0.7)  # Variation de la pression en arrière-plan
fig.colorbar(contour, ax=ax, label='Pression P')

# Taille du sous-échantillonnage
sampling_step = 3  # Ajustez cette valeur pour contrôler la densité des flèches

# Sous-échantillonnage des données
Ux_subsampled = Ux[::sampling_step, ::sampling_step]
Uy_subsampled = Uy[::sampling_step, ::sampling_step]

# Sous-échantillonnage de la grille
X_grid_subsampled = X_grid[::sampling_step, ::sampling_step]
Y_grid_subsampled = Y_grid[::sampling_step, ::sampling_step]



# Contour pour la pression
contour = ax.contourf(Y_grid, X_grid, P, cmap='viridis', alpha=0.7)
fig.colorbar(contour, ax=ax, label='Pression P')

# Tracer le quiver avec les données sous-échantillonnées
ax.quiver(Y_grid_subsampled, X_grid_subsampled, Ux_subsampled, Uy_subsampled, scale_units='xy', angles='xy', color='blue')
ax.set_title('Quiver plot: Vecteurs de vitesse (Ux, Uy) sous-échantillonnés')
ax.set_xlabel('Index Y')
ax.set_ylabel('Index X')



# ==============================================
# 1.COMPARAISSON A POISSEUILLE 
# ==============================================
import numpy as np
import matplotlib.pyplot as plt

# Choix de l'indice i pour la coupe en X (vertical)
i_center = M // 2  # milieu en X
Ux_profile = Ux[i_center, :]  # extraire Ux sur la ligne i_center

# Coordonnées réelles en Y (ou j indices)
y_coords = np.linspace(0, l, N)  # si on veut l'échelle physique en mètres

plt.figure(figsize=(8,5))
plt.plot(y_coords, Ux_profile, label='Ux Numérique')
plt.xlabel('Y (m)')
plt.ylabel('Ux (m/s)')
plt.title('Coupe de Ux au milieu du domaine (i={})'.format(i_center))

# ==============================================
# Option : comparer à une solution Poiseuille 2D analytique
# (canal de hauteur l, deltaP imposé, etc.)
# ==============================================
# Formule simplifiée si on considère un canal horizontal hauteur l
# et un gradient de pression linéaire entre P1 et P2, viscosité e.
# Hypothèse : P1>P2, paroi y=0 et y=l => Ux=0
# => Ux_ana(y) ~ (1/(2*mu * Lx)) * dP * ( y*(l-y) ) 
# (à ajuster à vos constantes, c'est un exemple.)

DeltaP = (P1 - P2)  # pression imposée 
mu = e               #  "viscosité" 
Lx = L               # longueur dans x 

y = np.linspace(0, l, N)  
# Formule "Poiseuille" hypothétique :
Ux_ana = (DeltaP / (2*mu * Lx)) * (y*(l - y))  

plt.plot(y, Ux_ana, '--r', label='Ux Poiseuille (analytique)')
plt.legend()
plt.grid(True)


# ==============================================
# 2. ERREUR POINT PAR POINT (2D)
# ==============================================
Ux_ana_2D = np.zeros_like(Ux)
for i in range(M):
    for j in range(N):
        # Convertir (i,j) -> y réel
        y_val = j * Dy  # ou plus précisément y0 + j*Dy si y0=0
        # Formule Poiseuille en tout point (si le flux est parfaitement 1D en y)
        Ux_ana_2D[i,j] = (DeltaP / (2*mu*Lx)) * (y_val*(l - y_val))

# Erreur : Ux_num - Ux_ana
error_2D = Ux - Ux_ana_2D


plt.figure(figsize=(7,6))
plt.imshow(error_2D, cmap='bwr', aspect='auto')
plt.colorbar(label='Erreur Ux (num - ana)')
plt.title('Erreur point par point de Ux')
plt.xlabel('Index Y')
plt.ylabel('Index X')




# ==============================================
# 3. ERREUR GLOBALE
# ==============================================
diff = Ux - Ux_ana_2D
err_L2 = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(Ux_ana_2D**2))
print("Erreur L2 relative sur Ux =", err_L2)

# ==============================================
# 4. DEBIT
# ==============================================
# Prenons une coupe verticale par exemple à x = i_center (même i_center).
# Débit Q ~ ∫ Ux dy (pour un canal 2D d'épaisseur 1 m ou si on considère l'unité en z=1)
# Approche discrète : somme(Ux(i_center,j)) * Dy
Q_num = np.sum(Ux_profile) * Dy

# Débit analytique
# Q_ana = ∫(0..l) Ux_ana(y) dy
# Si Ux_ana(y) = (DeltaP / (2*mu*Lx)) * [ y(l-y) ], vous pouvez l'intégrer analytiquement
# L'intégrale de y(l-y) dy de 0..l = l^3 / 6
Q_ana = (DeltaP / (2*mu*Lx)) * (l**3 / 6)

print("Débit numérique =", Q_num)
print("Débit analytique (Poiseuille) =", Q_ana)
print("Erreur relative sur le débit =", abs(Q_num - Q_ana)/Q_ana)


# ==============================================
# 5. Residut
# ==============================================


# Juste après avoir X = spsolve(Z, Y):

R_vector = Z @ X - Y  # vecteur résidu global

# Reshape par bloc
resUx = R_vector[:M*N].reshape((M,N))
resUy = R_vector[M*N:2*M*N].reshape((M,N))
resP  = R_vector[2*M*N:].reshape((M,N))



resUx_rel = np.zeros((M, N))
resUy_rel = np.zeros((M, N))
resP_rel  = np.zeros((M, N))


#résidu Y

# Construisons le résidu relatif mixte pour eq. (Uy)
resUy_rel_mix = np.zeros((M,N))
eps = 1e-14

for i in range(1, M-1):
    for j in range(1, N-1):
        denom = np.linalg.norm(Y)
        if denom < eps:
            denom = eps
        resUy_rel_mix[i,j] = abs(resUy[i,j])  / denom

# Visualisation
plt.figure(figsize=(6,5))
im = plt.imshow(resUy_rel_mix, cmap='bwr', aspect='auto')
plt.colorbar(im, label='Résidu relatif mixte eq. Uy')
plt.title("Résidu relatif eq. (Uy) = |resUy| / [ |Y]")
plt.xlabel("Index j (Y)")
plt.ylabel("Index i (X)")



#résidu X

resUx_rel_mix = np.zeros((M,N))
eps = 1e-14

for i in range(1,M-1):
    for j in range(1,N-1):
        denom = np.linalg.norm(Y)
        if denom < eps:
            denom = eps
        resUx_rel_mix[i,j] = abs(resUx[i,j])  / denom

plt.figure()
im = plt.imshow(resUx_rel_mix, cmap='bwr', aspect='auto')
plt.colorbar(im, label='Résidu relatif mixte eq. Ux')
plt.title("Résidu relatif eq. (Ux) = |resUx| / [|Y|]")
plt.xlabel("Index j (Y)")
plt.ylabel("Index i (X)")


#résidu P
resP_rel_mix = np.zeros((M,N))
eps = 1e-14

for i in range(1,M-1):
    for j in range(1,N-1):
        denom = np.linalg.norm(Y)
        if denom < eps:
            denom = eps
        resP_rel_mix[i,j] = abs(resP[i,j])  / denom

plt.figure()
im = plt.imshow(resP_rel_mix, cmap='bwr', aspect='auto')
plt.colorbar(im, label='Résidu relatif mixte eq. P')
plt.title("Résidu relatif eq. (P) = |resP| / [|Y|]")
plt.xlabel("Index j (Y)")
plt.ylabel("Index i (X)")
plt.show()



