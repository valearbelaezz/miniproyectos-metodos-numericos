import numpy as np
import matplotlib.pyplot as plt

def main():

    # Parameters and constants
    Lx = 8
    Ly = 6
    n_nodos_y = 50
    n_nodos_x = 100
    Vp1 = 4
    Vp2 = -2
    max_iter = 1000
    dx = Lx / (n_nodos_x - 1)
    dy = Ly / (n_nodos_y - 1)

    # mesh grid
    x = np.linspace(0, Lx, n_nodos_x)
    y = np.linspace(0, Ly, n_nodos_y)

    # create array for potential values
    V = np.zeros((n_nodos_x, n_nodos_y))

    # index for y=2 and y=4
    j1 = np.argmin(np.abs(y - 2))
    j2 = np.argmin(np.abs(y - 4))

    # index for x=2 and x=6
    i_min = np.argmin(np.abs(x - 2))
    i_max = np.argmin(np.abs(x - 6))

    # set Dirichlet conditions for plates (only between x=2 and x=6)
    V[i_min:i_max+1, j1] = Vp1
    V[i_min:i_max+1, j2] = Vp2

    # solution by relaxation jacobi method
    for it in range(max_iter):
        V_new = V.copy()
        # update only interior points (no borders or plates)
        for i in range(1, n_nodos_x - 1):
            for j in range(1, n_nodos_y - 1):
                # saltar donde están las placas
                if (j == j1 or j == j2) and (i_min <= i <= i_max):
                    continue
                V_new[i, j] = (dy**2 * (V[i+1, j] + V[i-1, j]) +
                               dx**2 * (V[i, j+1] + V[i, j-1])) / (2 * (dx**2 + dy**2))

        # border conditions (caja a 0)
        V_new[0, :] = 0
        V_new[-1, :] = 0
        V_new[:, 0] = 0
        V_new[:, -1] = 0

        # reimpose Dirichlet conditions for plates
        V_new[i_min:i_max+1, j1] = Vp1
        V_new[i_min:i_max+1, j2] = Vp2

        V = V_new

    return V, x, y, j1, j2, i_min, i_max


# run solver
V, x, y, j1, j2, i_min, i_max = main()

# plot
X, Y = np.meshgrid(x, y, indexing="ij")  # correct meshgrid
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, V, 50, cmap="seismic")
plt.colorbar(contour, label="Potential (V)")

# contour lines
plt.contour(X, Y, V, 20, colors="black", linewidths=0.5, alpha=0.6)

# draw plates (solo entre x=2 y x=6)
plt.plot(x[i_min:i_max+1], [y[j1]]*(i_max-i_min+1), color="red", linewidth=3, label=f"Plate V={4} V")
plt.plot(x[i_min:i_max+1], [y[j2]]*(i_max-i_min+1), color="blue", linewidth=3, label=f"Plate V={-2} V")

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Potential Distribution in Capacitor (Restricted Plates, Jacobi Method)")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()



