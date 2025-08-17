import numpy as np
import matplotlib.pyplot as plt

#Parámetros iniciales
a = 1       # Constante de decaimiento
I = 1       # Condición inicial
t_max = 1    # Tiempo máximo de simulación
dt_values = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]  # Diferentes Δt para probar

#Sln exacta
def exact_solution(t):
    return I * np.exp(-a * t)

#Métodos numéricos
def forward_euler(dt):
    n_steps = int(t_max / dt)
    u = np.zeros(n_steps + 1)
    u[0] = I
    for i in range(n_steps):
        u[i + 1] = u[i] - a * u[i] * dt
    t = np.linspace(0, t_max, n_steps + 1)
    return t, u

def crank_nicolson(dt):
    n_steps = int(t_max / dt)
    u = np.zeros(n_steps + 1)
    u[0] = I
    for i in range(n_steps):
        u[i + 1] = (1 - a * dt / 2) / (1 + a * dt / 2) * u[i]
    t = np.linspace(0, t_max, n_steps + 1)
    return t, u

def backward_euler(dt):
    n_steps = int(t_max / dt)
    u = np.zeros(n_steps + 1)
    u[0] = I
    for i in range(n_steps):
        u[i + 1] = u[i] / (1 + a * dt)
    t = np.linspace(0, t_max, n_steps + 1)
    return t, u

#Graficar soluciones exactas y aproximadas
def plot_comparison(dt_values):
    t_exact = np.linspace(0, t_max, 1000)
    u_exact = exact_solution(t_exact)

    for dt in dt_values:
        t_fe, u_fe = forward_euler(dt)
        t_cn, u_cn = crank_nicolson(dt)
        t_be, u_be = backward_euler(dt)

        plt.figure(figsize=(10,6))
        plt.plot(t_exact, u_exact, label='Solución Exacta', color='blue')
        plt.plot(t_fe, u_fe, '--', label=f'Forward Euler (Δt={dt})', color='red')
        plt.plot(t_cn, u_cn, '-.', label=f'Crank-Nicolson (Δt={dt})', color='green')
        plt.plot(t_be, u_be, ':', label=f'Backward Euler (Δt={dt})', color='purple')
        plt.xlabel('Tiempo (t)')
        plt.ylabel('u(t)')
        plt.title('Solución Exacta vs Aproximaciones')
        plt.grid(True)
        plt.legend()
        plt.show()

#Calcular errores máximos
def compute_errors(dt_values):
    errors_fe = []
    errors_cn = []
    errors_be = []

    for dt in dt_values:
        t_fe, u_fe = forward_euler(dt)
        t_cn, u_cn = crank_nicolson(dt)
        t_be, u_be = backward_euler(dt)

        errors_fe.append(np.max(np.abs(u_fe - exact_solution(t_fe))))
        errors_cn.append(np.max(np.abs(u_cn - exact_solution(t_cn))))
        errors_be.append(np.max(np.abs(u_be - exact_solution(t_be))))

    return np.array(errors_fe), np.array(errors_cn), np.array(errors_be)


#Graficar error máximo vs Δt
def plot_error(dt_values, errors_fe, errors_cn, errors_be):
    plt.figure(figsize=(10,6))
    plt.plot(dt_values, errors_fe, 'o-', label='Forward Euler')
    plt.plot(dt_values, errors_cn, 'x-', label='Crank-Nicolson')
    plt.plot(dt_values, errors_be, 's-', label='Backward Euler')
    plt.xlabel('Δt')
    plt.ylabel('Error máximo')
    plt.title('Error máximo vs Δt')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


# Graficar log-log del error y calcular pendiente
def plot_loglog_and_slope(dt_values, errors_fe, errors_cn, errors_be):
    log_dt = np.log(dt_values)
    log_errors_fe = np.log(errors_fe)
    log_errors_cn = np.log(errors_cn)
    log_errors_be = np.log(errors_be)

    # Ajuste lineal para obtener pendientes
    p_fe, _ = np.polyfit(log_dt, log_errors_fe, 1)
    p_cn, _ = np.polyfit(log_dt, log_errors_cn, 1)
    p_be, _ = np.polyfit(log_dt, log_errors_be, 1)

    print(f"Orden de convergencia aproximado:")
    print(f"Forward Euler: {p_fe:.2f}")
    print(f"Crank-Nicolson: {p_cn:.2f}")
    print(f"Backward Euler: {p_be:.2f}")

    plt.figure(figsize=(10,6))
    plt.plot(log_dt, log_errors_fe, 'o-', label=f'Forward Euler, p~{p_fe:.2f}')
    plt.plot(log_dt, log_errors_cn, 'x-', label=f'Crank-Nicolson, p~{p_cn:.2f}')
    plt.plot(log_dt, log_errors_be, 's-', label=f'Backward Euler, p~{p_be:.2f}')
    plt.xlabel('log(Δt)')
    plt.ylabel('log(Error máximo)')
    plt.title('Relación log-log Error vs Δt')
    plt.grid(True)
    plt.legend()
    plt.show()


# Main
# 1. Graficar soluciones
plot_comparison(dt_values)

# 2. Calcular errores
errors_fe, errors_cn, errors_be = compute_errors(dt_values)

# 3. Graficar errores
plot_error(dt_values, errors_fe, errors_cn, errors_be)

# 4. Graficar log-log y obtener pendientes
plot_loglog_and_slope(dt_values, errors_fe, errors_cn, errors_be)
