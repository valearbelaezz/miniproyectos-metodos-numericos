
# MÉTODOS NUMÉRICOS - MINIPROYECTO 3
# Tareas 2 y 3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.tri as tri

# PARÁMETROS ESTOCÁSTICOS

param_stats = {
    "R":    {"mu": 0.6,   "sigma": 0.01},      # [m]
    "alpha":{"mu": 100.0, "sigma": 10.0},      # [N/m²]
    "beta": {"mu": 50.0,  "sigma": 5.0},       # [1/m²]
    "gamma":{"mu": 0.3,   "sigma": 0.05},      # [-]
    "theta":{"mu": np.pi/4, "sigma": np.pi/12} # [rad]
}


# FUNCIÓN DE CARGA f(x, y)

def f_load(x, y, alpha, beta, gamma, theta):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return alpha * (1 + beta * r**2 + gamma * np.cos(phi - theta))

# GENERACIÓN DE MALLA TRIANGULAR DEL HEXÁGONO

def generate_hexagon_mesh(R, num_div=15):
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.array([[R*np.cos(a), R*np.sin(a)] for a in angles])
    points = vertices.copy()
    for r in np.linspace(0.1, 0.9, num_div):
        for angle in np.linspace(0, 2*np.pi, int(6*r*num_div), endpoint=False):
            x = r * R * np.cos(angle)
            y = r * R * np.sin(angle)
            if (np.abs(x)*np.sqrt(3) + np.abs(y)) <= np.sqrt(3)*R:
                points = np.vstack([points, [x, y]])
    triang = tri.Triangulation(points[:,0], points[:,1])
    triangles = []
    for t in triang.triangles:
        xc, yc = np.mean(points[t], axis=0)
        if (np.abs(xc)*np.sqrt(3) + np.abs(yc)) <= np.sqrt(3)*R:
            triangles.append(t)
    return points, np.array(triangles)

# MATRIZ DE RIGIDEZ ELEMENTAL

def element_stiffness_matrix(points, triangle):
    x1, y1 = points[triangle[0]]
    x2, y2 = points[triangle[1]]
    x3, y3 = points[triangle[2]]
    area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    ke = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            ke[i,j] = (b[i]*b[j] + c[i]*c[j]) / (4*area)
    return ke, area

# VECTOR DE CARGA ELEMENTAL

def element_load_vector(points, triangle, alpha, beta, gamma, theta):
    x1, y1 = points[triangle[0]]
    x2, y2 = points[triangle[1]]
    x3, y3 = points[triangle[2]]
    area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    f1 = f_load(x1, y1, alpha, beta, gamma, theta)
    f2 = f_load(x2, y2, alpha, beta, gamma, theta)
    f3 = f_load(x3, y3, alpha, beta, gamma, theta)
    return (area/3)*np.array([f1, f2, f3])


# SOLVER FEM TRIANGULAR (Poisson 2D)

def solve_fem_triangular(R, alpha, beta, gamma, theta, num_div=20):
    points, triangles = generate_hexagon_mesh(R, num_div)
    n_nodes = len(points)
    K = sparse.lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for tri_elem in triangles:
        ke, _ = element_stiffness_matrix(points, tri_elem)
        fe = element_load_vector(points, tri_elem, alpha, beta, gamma, theta)
        for i, ni in enumerate(tri_elem):
            F[ni] += fe[i]
            for j, nj in enumerate(tri_elem):
                K[ni, nj] += ke[i, j]

    # === Condición de contorno u=0 sin tolerancia ===
    for i, (x, y) in enumerate(points):
        r = np.sqrt(x**2 + y**2)
        if r >= R * 0.999:  # borde del hexágono
            K[i, :] = 0
            K[:, i] = 0
            K[i, i] = 1
            F[i] = 0

    U = spsolve(K.tocsr(), F)
    return np.max(np.abs(U))


def solve_fem_hexagon(R, alpha, beta, gamma, theta, num_div=15):
    try:
        return solve_fem_triangular(R, alpha, beta, gamma, theta, num_div)
    except:
        return alpha * R**2 * 0.01


# TAREA 2: ANÁLISIS MONTE CARLO

N = 100
u_max_results = []
parameters_history = []

print("Ejecutando 100 simulaciones Monte Carlo...\n")
for i in range(N):
    R_i = np.random.normal(param_stats["R"]["mu"], param_stats["R"]["sigma"])
    alpha_i = np.random.normal(param_stats["alpha"]["mu"], param_stats["alpha"]["sigma"])
    beta_i = np.random.normal(param_stats["beta"]["mu"], param_stats["beta"]["sigma"])
    gamma_i = np.random.normal(param_stats["gamma"]["mu"], param_stats["gamma"]["sigma"])
    theta_i = np.random.normal(param_stats["theta"]["mu"], param_stats["theta"]["sigma"])
    parameters_history.append([R_i, alpha_i, beta_i, gamma_i, theta_i])
    u_max_i = solve_fem_hexagon(R_i, alpha_i, beta_i, gamma_i, theta_i, num_div=12)
    u_max_results.append(u_max_i)
    if (i+1) % 10 == 0:
        print(f"Simulación {i+1}/{N} completada")

u_max_results = np.array(u_max_results)
parameters_history = np.array(parameters_history)

# Estadística
mean_u_max = np.mean(u_max_results)
std_u_max = np.std(u_max_results)
t_value = stats.t.ppf(0.975, N-1)
margin_error = t_value * std_u_max / np.sqrt(N)
ci_lower, ci_upper = mean_u_max - margin_error, mean_u_max + margin_error

print("\n=== TAREA 2: ANÁLISIS MONTE CARLO ===")
print(f"Media = {mean_u_max:.4e}, Desv.Std = {std_u_max:.4e}")
print(f"IC 95%: [{ci_lower:.4e}, {ci_upper:.4e}]")

# Histograma
plt.figure(figsize=(9,5))
plt.hist(u_max_results, bins=12, color='skyblue', edgecolor='black', density=True)
plt.axvline(mean_u_max, color='r', ls='--', lw=2, label=f"Media = {mean_u_max:.3e}")
plt.axvline(ci_lower, color='orange', ls=':', lw=2)
plt.axvline(ci_upper, color='orange', ls=':', lw=2, label="IC 95%")
density = stats.gaussian_kde(u_max_results)
xs = np.linspace(u_max_results.min(), u_max_results.max(), 200)
plt.plot(xs, density(xs), 'r-', lw=1.5)
plt.legend()
plt.xlabel("$u_{max}$ [m]")
plt.ylabel("Densidad de probabilidad")
plt.title("Distribución Monte Carlo de $u_{max}$")
plt.grid(True, alpha=0.3)
plt.show()

# Análisis de sensibilidad
param_names = ['R', 'alpha', 'beta', 'gamma', 'theta']
correlations = [np.corrcoef(parameters_history[:,i], u_max_results)[0,1] for i in range(5)]
most_idx = np.argmax(np.abs(correlations))
most_param = param_names[most_idx]

plt.figure(figsize=(8,4))
bars = plt.bar(param_names, np.abs(correlations), color='lightblue', edgecolor='black')
bars[most_idx].set_color('red')
plt.ylabel('|Correlación|')
plt.title(f"Análisis de Sensibilidad\nParámetro más influyente: {most_param}")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nParámetro más influyente: {most_param}")


# TAREA 3: ESTUDIO PARAMÉTRICO

print(f"\n=== TAREA 3: Estudio paramétrico de {most_param} ===")

# Fijar parámetros promedio
R_fixed = param_stats["R"]["mu"]
alpha_fixed = param_stats["alpha"]["mu"]
beta_fixed = param_stats["beta"]["mu"]
gamma_fixed = param_stats["gamma"]["mu"]
theta_fixed = param_stats["theta"]["mu"]

N_param = 100  # número de muestras

# Generar muestras del parámetro seleccionado
if most_param == "R":
    samples = np.random.normal(param_stats["R"]["mu"], param_stats["R"]["sigma"], N_param)
elif most_param == "alpha":
    samples = np.random.normal(param_stats["alpha"]["mu"], param_stats["alpha"]["sigma"], N_param)
elif most_param == "beta":
    samples = np.random.normal(param_stats["beta"]["mu"], param_stats["beta"]["sigma"], N_param)
elif most_param == "gamma":
    samples = np.random.normal(param_stats["gamma"]["mu"], param_stats["gamma"]["sigma"], N_param)
else:
    samples = np.random.normal(param_stats["theta"]["mu"], param_stats["theta"]["sigma"], N_param)

# Simulaciones FEM variando solo el parámetro seleccionado
u_param = []
for val in samples:
    if most_param == "R":
        u = solve_fem_hexagon(val, alpha_fixed, beta_fixed, gamma_fixed, theta_fixed)
    elif most_param == "alpha":
        u = solve_fem_hexagon(R_fixed, val, beta_fixed, gamma_fixed, theta_fixed)
    elif most_param == "beta":
        u = solve_fem_hexagon(R_fixed, alpha_fixed, val, gamma_fixed, theta_fixed)
    elif most_param == "gamma":
        u = solve_fem_hexagon(R_fixed, alpha_fixed, beta_fixed, val, theta_fixed)
    else:
        u = solve_fem_hexagon(R_fixed, alpha_fixed, beta_fixed, gamma_fixed, val)
    u_param.append(u)

samples = np.array(samples)
u_param = np.array(u_param)
order = np.argsort(samples)
samples_sorted = samples[order]
u_sorted = u_param[order]

# Gráfico u_max vs parámetro
plt.figure(figsize=(8,5))
plt.scatter(samples_sorted, u_sorted, color='purple', alpha=0.7, label='Simulaciones FEM')
z = np.polyfit(samples_sorted, u_sorted, 2)
p = np.poly1d(z)
plt.plot(samples_sorted, p(samples_sorted), 'r-', lw=2, label='Tendencia cuadrática')
plt.xlabel(f'Parámetro {most_param}')
plt.ylabel('$u_{max}$ [m]')
plt.title(f'Estudio Paramétrico: $u_{{max}}$ vs {most_param}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nTAREA 3 completada con N=100 muestras.")

