from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

import numpy as np
import os

# Paramètres
DX = 0.2
C = 4.7 * 10 ** -6
L = 3
D = 19.8
C_0 = 9.28
lambdaa = 300
r=-lambdaa/(DX+lambdaa)
bordure_dirichlet = []   

def discretize_space(rectangle, step_size):
    # Définir les limites du rectangle initial
    x_min, y_min, x_max, y_max = rectangle
    
    # Créer une grille discrète
    x_values = list(range(int(x_min), int(x_max) + 1, step_size))
    y_values = list(range(int(y_min), int(y_max) + 1, step_size))

    # Créer une liste pour stocker les points de la figure résultante
    figure_points = []
 
    # Ajouter les points du rectangle initial à la liste
    for x in x_values:
        for y in y_values:
            figure_points.append((x, y))

    
    # Créer un dictionnaire pour stocker les voisins de chaque point
    neighbors_dict = {}
   
    # Trouver les voisins de chaque point
    for i, point in enumerate(figure_points, start=1):
        neighbors = []

        x, y = point
       
        # Voisin à gauche
        left_neighbor = (x - step_size, y)
        if left_neighbor in figure_points:
            neighbors.append(figure_points.index(left_neighbor) + 1)
        else:
            neighbors.append(0)
       
        # Voisin à droite
        right_neighbor = (x + step_size, y)
        if right_neighbor in figure_points:
            neighbors.append(figure_points.index(right_neighbor) + 1)
        else:
            neighbors.append(0)
       
        # Voisin en dessous
        below_neighbor = (x, y - step_size)
        if below_neighbor in figure_points:
            neighbors.append(figure_points.index(below_neighbor) + 1)
        else:
            neighbors.append(0)
       
        # Voisin au-dessus
        above_neighbor = (x, y + step_size)
        if above_neighbor in figure_points:
            neighbors.append(figure_points.index(above_neighbor) + 1)
        else:
            neighbors.append(0)
        
        
        neighbors_dict[i] = neighbors
    # Calculer les coefficients pour chaque point
    coefficients_dict = {}
    for i, point in enumerate(figure_points, start=1):
        x, y = point
        k1, k2, k3, k4 = neighbors_dict[i]
        k_neighbors = [n for n in neighbors_dict[i] if n != 0]# exclus les voisins non existant
        coefficients_dict[i] = {
            'k1': 1 if k1 != 0 else 0,
            'k2': 1 if k2 != 0 else 0,
            'k3': 1 if k3 != 0 else 0,
            'k4': 1 if k4 != 0 else 0,
            'k': -len(k_neighbors)
        }

    return figure_points, neighbors_dict, coefficients_dict

def assign_boundary_conditions(rectangle, step_size):

   
    # Create a dictionary to store boundary conditions for each point
    boundary_conditions = {}

    for point in [(0, y) for y in range(rectangle[1], rectangle[3] + 1, step_size)]:
        boundary_conditions[point] = 1 
   
    for point in [(rectangle[2], y) for y in range(rectangle[1], rectangle[3] + 1, step_size)]:
        boundary_conditions[point] = 2  # Dirichelt condition
    for point in bordure_dirichlet:
        print(point)
        boundary_conditions[point] = 2
    return boundary_conditions
    
def build_matrix_B(coefficients_dict, neighbors_dict, boundary_conditions, figure_points, alpha):
    num_points = len(coefficients_dict)
    matrix_B = np.zeros((num_points, num_points))
    matrix_C = np.zeros((num_points, num_points))
    
    for k in range(1, num_points + 1):
        matrix_B[k - 1, k - 1] = 1 - coefficients_dict[k]['k'] * alpha
        matrix_C[k - 1, k - 1] = 1 + coefficients_dict[k]['k'] * alpha
        
        for idx, coef in enumerate(['k1', 'k2', 'k3', 'k4']):
            if neighbors_dict[k][idx] > 0:
                matrix_B[k - 1, neighbors_dict[k][idx] - 1] = -coefficients_dict[k][coef] * alpha
                matrix_C[k - 1, neighbors_dict[k][idx] - 1] = coefficients_dict[k][coef] * alpha
                
        condition_k = boundary_conditions.get(figure_points[k - 1], 0)
        if condition_k == 1:
            matrix_B[k - 1, :] = 0
            matrix_C[k - 1, :] = 0
            matrix_B[k - 1, k - 1] = 1  
            matrix_C[k - 1, k - 1] = 1 
        if condition_k == 2 and coefficients_dict[k]['k']!=-4 and coefficients_dict[k]['k']!=-2 :
                matrix_B[k - 1, neighbors_dict[k][0] - 1] = r * (coefficients_dict[k]['k1'] * coefficients_dict[k]['k3'] * coefficients_dict[k]['k4'])
                matrix_C[k - 1, neighbors_dict[k][0] - 1] = r * (coefficients_dict[k]['k1'] * coefficients_dict[k]['k3'] * coefficients_dict[k]['k4'])

                matrix_B[k - 1, neighbors_dict[k][1] - 1] = r * (coefficients_dict[k]['k2'] * coefficients_dict[k]['k3'] * coefficients_dict[k]['k4'])
                matrix_C[k - 1, neighbors_dict[k][1] - 1] = r * (coefficients_dict[k]['k2'] * coefficients_dict[k]['k3'] * coefficients_dict[k]['k4'])

                matrix_B[k - 1, neighbors_dict[k][2] - 1] = r * (coefficients_dict[k]['k3'] * coefficients_dict[k]['k2'] * coefficients_dict[k]['k1'])
                matrix_C[k - 1, neighbors_dict[k][2] - 1] = r * (coefficients_dict[k]['k3'] * coefficients_dict[k]['k2'] * coefficients_dict[k]['k1'])

                matrix_B[k - 1, neighbors_dict[k][3] - 1] = r * (coefficients_dict[k]['k4'] * coefficients_dict[k]['k2'] * coefficients_dict[k]['k1'])
                matrix_C[k - 1, neighbors_dict[k][3] - 1] = r * (coefficients_dict[k]['k4'] * coefficients_dict[k]['k2'] * coefficients_dict[k]['k1'])

                matrix_B[k - 1, k - 1] = 1 
                matrix_C[k - 1, k - 1] = 1 
                
        
                
        if condition_k == 2 and coefficients_dict[k]['k']==-2 :
                matrix_B[k - 1, neighbors_dict[k][0] - 1] = r *coefficients_dict[k]['k1']
                matrix_C[k - 1, neighbors_dict[k][0] - 1] = r *coefficients_dict[k]['k1']


                matrix_B[k - 1, neighbors_dict[k][1] - 1] = r *coefficients_dict[k]['k2']
                matrix_C[k - 1, neighbors_dict[k][1] - 1] = r *coefficients_dict[k]['k2']


                matrix_B[k - 1, neighbors_dict[k][2] - 1] = 0
                matrix_C[k - 1, neighbors_dict[k][2] - 1] = 0


                matrix_B[k - 1, neighbors_dict[k][3] - 1] = 0
                matrix_C[k - 1, neighbors_dict[k][3] - 1] = 0


                matrix_B[k - 1, k - 1] = 1 
                matrix_C[k - 1, k - 1] = 1 

    return matrix_B, matrix_C
       
class DiffusionAnimation:
    def __init__(self, matrix_B, matrix_C, X0, figure_points):

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.matrix_B = matrix_B
        self.matrix_C = matrix_C
        self.X0 = X0
        self.prev_Xn = self.X0
        self.figure_points = figure_points
        self.x_values, self.y_values = zip(*figure_points)
        self.unique_x = sorted(set(self.x_values))
        self.unique_y = sorted(set(self.y_values))
        
        # Calculer la matrice de transition une seule fois
        self.inverse_matrix_B = np.linalg.inv(self.matrix_B)
        self.transition_matrix = np.dot(self.inverse_matrix_B, self.matrix_C)
        
        self.text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, color='white', bbox=dict(facecolor='black'))
        self.fig.canvas.mpl_connect('motion_notify_event', self.show_concentration)

        # Initialiser le plot
        self.Q = np.zeros((len(self.unique_y), len(self.unique_x)))
        for idx, (x, y) in enumerate(self.figure_points):
            i, j = self.unique_y.index(y), self.unique_x.index(x)
            self.Q[i, j] = X0[idx]
        self.mesh = self.ax.pcolormesh(
            np.array(self.unique_x),
            np.array(self.unique_y),
            self.Q,
            cmap='inferno',
            shading='auto'
        )
        plt.title("Diffusion pulmonaire dans le régime temporel")
        self.colorbar = plt.colorbar(self.mesh, ax=self.ax,label='Concentration (mmol/l)')
       
        self.ax.set_xlabel('Longueur (10^-2 mm)')
        self.ax.set_ylabel('Longueur (10^-2 mm)')
    def show_concentration(self, event):
        x, y = None, None  # Valeurs par défaut pour éviter l'erreur
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata  # Définir x et y uniquement si event.inaxes est valide
        
        if x is not None and y is not None:
            try:
                # Trouver le point de grille le plus proche
                x_closest = min(self.unique_x, key=lambda xi: abs(xi - x))
                y_closest = min(self.unique_y, key=lambda yi: abs(yi - y))
                
                # Trouver la concentration correspondante
                i = self.unique_y.index(y_closest)
                j = self.unique_x.index(x_closest)
                concentration = self.Q[i, j]
                
                # Mettre à jour le texte affiché
                self.text.set_text(f'Concentration: {concentration:.4e} mmol/l')
                self.text.set_position((event.xdata, event.ydata))
                self.fig.canvas.draw_idle()
            except ValueError:
                pass

    
    def update(self, frame):
        # Calcul de la nouvelle concentration
        
        Xn = np.dot(self.transition_matrix, self.prev_Xn)
        self.prev_Xn = Xn
        # Mise à jour de la matrice Q
        for idx, (x, y) in enumerate(self.figure_points):
            i, j = self.unique_y.index(y), self.unique_x.index(x)
            self.Q[i, j] = Xn[idx]
        
        # Mise à jour des limites du color bar
        self.mesh.set_clim(vmin=np.min(self.Q), vmax=np.max(self.Q))
        self.colorbar.update_normal(self.mesh)  # Synchronise la colorbar avec les nouvelles limites
        
        # Mise à jour du plot
        self.mesh.set_array(self.Q.ravel())
        plt.title(f'Itération = {frame}')
        return [self.mesh]


def main():
    # Paramètres de simulation
    rectangle = (0, 0, 300, 300)
    step_size = 20
    dt = 1/144
    alpha = dt * D / (2 * ((step_size*0.01)**2))

    # Initialisation
    #figure_points, neighbors_dict, coefficients_dict = discretize_space(rectangle, step_size)
    #boundary_conditions = assign_boundary_conditions(rectangle, step_size)
    figure_points, neighbors_dict, coefficients_dict = discretize_space(rectangle, step_size)
    boundary_conditions = assign_boundary_conditions(rectangle, step_size)
    matrix_B, matrix_C = build_matrix_B(coefficients_dict, neighbors_dict, boundary_conditions, figure_points, alpha)
   
    # Conditions initiales
    X0 = np.zeros(len(figure_points))
    for i, point in enumerate(figure_points):
        condition = boundary_conditions.get(point, 0)
        X0[i] = C_0 if condition == 1 else 0
    # Création de l'animation
    anim_object = DiffusionAnimation(matrix_B, matrix_C, X0, figure_points)
    anim = FuncAnimation(
        anim_object.fig,
        anim_object.update,
        frames=300,  
        interval=1000/144,  
        blit=False,
        repeat=False
    )
    plt.show()
    # Save the animation as GIF using PillowWriter
    writer = PillowWriter(fps=20)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    anim.save(os.path.join(output_dir, "diffusion_transient.gif"), writer=writer)
    plt.close()

if __name__ == "__main__":
    main()