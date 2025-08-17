import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

# ---------------------------
# 1) Datos e interpolación
# ---------------------------

# Puntos de muestreo (x, y)
puntos = np.array([
    (2.75, 0.00), (3.72, 6.06), (7.99, 4.96), (10.00, 5.24), (13.98, 6.45),
    (15.66, 6.59), (20.78, 4.52), (22.78, 0.00), (19.03, 6.01)
], dtype=float)

x = puntos[:, 0]
y = puntos[:, 1]
n = len(x)

# Matriz de Vandermonde para polinomio con base {1, x, x^2, ...}
A = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        A[i, j] = x[i]**j

# Coeficientes c del polinomio p(x) = c0 + c1 x + c2 x^2 + ... + c8 x^8
c = np.linalg.solve(A, y)

# Evaluador del polinomio usando Horner (más estable numéricamente)
def p_eval(xv):
    xv = np.asarray(xv, dtype=float)
    # Horner requiere los coeficientes de mayor a menor grado
    coeffs = c[::-1]
    out = np.zeros_like(xv, dtype=float)
    for a in coeffs:
        out = out * xv + a
    return out

# ---------------------------
# 2) Volumen de revolución V = pi * ∫ [p(x)]^2 dx
# ---------------------------

a, b = float(np.min(x)), float(np.max(x))

# ---- Método A: Simpson compuesto (implementado) ----
def simpson_compuesto(fun, a, b, N=1000):
    """
    Simpson compuesto para integrar fun en [a,b].
    N debe ser par. Si no lo es, se incrementa en 1.
    """
    if N % 2 == 1:
        N += 1
    xs = np.linspace(a, b, N + 1)
    fx = fun(xs)
    h = (b - a) / N
    # Simpson: h/3 [f0 + fN + 4*(impares) + 2*(pares interiores)]
    S_odd = np.sum(fx[1:-1:2])
    S_even = np.sum(fx[2:-1:2])
    return (h / 3.0) * (fx[0] + fx[-1] + 4.0 * S_odd + 2.0 * S_even)

def integrando(xv):
    return p_eval(xv)**2

V_simpson = np.pi * simpson_compuesto(integrando, a, b, N=2000)  # malla densa

# ---- Método B: Cuadratura de Gauss-Legendre ----
def gauss_legendre(fun, a, b, n_nodes=32):
    """
    Integra fun en [a,b] con n_nodes puntos de Gauss-Legendre.
    """
    xi, wi = leggauss(n_nodes)               # nodos y pesos en [-1,1]
    # cambio de variable: x = (b-a)/2 * xi + (a+b)/2
    xm = 0.5 * (a + b)
    xr = 0.5 * (b - a)
    xg = xm + xr * xi
    return xr * np.sum(wi * fun(xg))

V_gauss = np.pi * gauss_legendre(integrando, a, b, n_nodes=40)

# ---------------------------
# 3) Errores relativos
# ---------------------------
V_exp = 2200.0  # ml = cm^3
err_simpson = abs((V_simpson - V_exp) / V_exp) * 100.0
err_gauss   = abs((V_gauss   - V_exp) / V_exp) * 100.0

print(f"Intervalo de integración: [{a:.2f}, {b:.2f}]")
print(f"Volumen (Simpson compuesto): {V_simpson:.3f} ml")
print(f"Error relativo (Simpson):    {err_simpson:.3f} %")
print()
print(f"Volumen (Gauss-Legendre):    {V_gauss:.3f} ml")
print(f"Error relativo (Gauss):      {err_gauss:.3f} %")

# ---------------------------
# 4) Gráfica de la interpolación
# ---------------------------
x_eval = np.linspace(a, b, 800)
y_eval = p_eval(x_eval)

plt.figure(figsize=(9,5))
plt.plot(x_eval, y_eval, label="Interpolación polinómica (grado 8)")
plt.scatter(x, y, color="red", zorder=3, label="Puntos de muestreo")
plt.fill_between(x_eval, 0, y_eval, alpha=0.15, label="Área que genera volumen (y^2)")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.title("Interpolación y volumen de revolución alrededor del eje x")
plt.xticks(np.arange(0, max(x_eval) + 5, 5))
plt.yticks(np.arange(max(0, np.min(y_eval) - 2), np.max(y_eval) + 2, 2))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
