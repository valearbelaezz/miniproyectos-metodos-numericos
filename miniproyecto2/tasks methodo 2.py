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
    max_iter = 5000
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

    # Dirichlet conditions
    V[:, j1] = Vp1
    V[:, j2] = Vp2

    # solution by relaxation jacobi method
    for it in range(max_iter):
        V_new = V.copy()
        # update only interior points (no borders or plates)
        for i in range(1, n_nodos_x - 1):
            for j in range(1, n_nodos_y - 1):
                if j in [j1, j2]:   # skip plates (fixed values)
                    continue
                V_new[i, j] = (dy**2 * (V[i+1, j] + V[i-1, j]) + dx**2 * (V[i, j+1] + V[i, j-1])) / (2 * (dx**2 + dy**2))

        # rewrite conditions in V array (border = 0)
        V_new[0, :] = 0     # y = 0
        V_new[-1, :] = 0    # y = Ly
        V_new[:, 0] = 0     # x = 0
        V_new[:, -1] = 0    # x = Lx

        # reapply Dirichlet conditions (plates)
        V_new[:, j1] = Vp1
        V_new[:, j2] = Vp2

        V = V_new

    return V, x, y, j1, j2


# run solver
V, x, y, j1, j2 = main()

# plot
X, Y = np.meshgrid(x, y, indexing="ij")  # correct meshgrid
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, V, 50, cmap="seismic")
plt.colorbar(contour, label="Potential (V)")

# contour lines
plt.contour(X, Y, V, 20, colors="black", linewidths=0.5, alpha=0.6)

# draw plates
plt.axhline(y[j1], color="red", linewidth=3, label=f"Plate V={4} V")
plt.axhline(y[j2], color="blue", linewidth=3, label=f"Plate V={-2} V")

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Potential Distribution in Capacitor (Jacobi Method)")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()

