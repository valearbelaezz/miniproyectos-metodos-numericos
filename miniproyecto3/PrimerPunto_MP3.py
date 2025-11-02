import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 1.0                     # Domain length
n_internal_nodes = 8        # Number of internal nodes
n_nodes = n_internal_nodes + 2  # Total nodes (including boundaries)

# Load function and analytical solution
def f(x):
    return np.sin(x)  # Source term f(x) = sin(x)

def u_exact(x):
    # Analytical solution for -u''(x) = sin(x), with u(0)=0 and u(1)=0
    # 1) Integrate twice: u(x) = sin(x) + C1*x + C2
    # 2) Apply BCs: u(0)=0 → C2=0,  u(1)=0 → sin(1) + C1 = 0 → C1 = -sin(1)
    # => u(x) = sin(x) - x*sin(1)
    return np.sin(x) - x * np.sin(1)

# Generate random mesh for FEM
# np.random.seed(42)  # Uncomment for reproducibility
internal_nodes = np.random.rand(n_internal_nodes)         
nodes = np.sort(np.concatenate(([0.0], internal_nodes, [L])))

print("Random mesh nodes:", nodes)

# Gaussian quadrature (2-point)
gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])   # Quadrature points
gauss_wts = np.array([1.0, 1.0])                      # Corresponding weights

# FEM assembly
K = np.zeros((n_nodes, n_nodes))   # Global stiffness matrix
F = np.zeros(n_nodes)              # Global load vector

for e in range(len(nodes)-1):
    x1, x2 = nodes[e], nodes[e+1]
    h_e = x2 - x1                  # Element length
    
    # Element stiffness matrix for linear elements
    Ke = (1.0/h_e) * np.array([[1, -1],
                               [-1, 1]])
    
    # Element load vector (Gaussian quadrature)
    Fe = np.zeros(2)
    for gp in range(len(gauss_pts)):
        xi = gauss_pts[gp]         # Local coordinate (-1 to 1)
        w = gauss_wts[gp]          # Quadrature weight
        
        # Map local coordinate to global x
        x = 0.5*(1 - xi)*x1 + 0.5*(1 + xi)*x2
        
        # Linear shape functions
        N1 = 0.5*(1 - xi)
        N2 = 0.5*(1 + xi)
        jacobian = h_e / 2.0
        
        # Add contribution to Fe
        Fe[0] += w * f(x) * N1 * jacobian
        Fe[1] += w * f(x) * N2 * jacobian
    
    # Assemble global matrices
    K[e:e+2, e:e+2] += Ke
    F[e:e+2] += Fe

# Apply boundary conditions (u(0)=u(1)=0)
K_bc = K[1:-1, 1:-1]
F_bc = F[1:-1]

# Solve linear system
u_internal = np.linalg.solve(K_bc, F_bc)

# Add boundary values
u_fem = np.concatenate(([0.0], u_internal, [0.0]))

# Plot results
x_fine = np.linspace(0, L, 200)

plt.figure(figsize=(10, 6))
plt.plot(x_fine, u_exact(x_fine), 'r-', label='Exact analytical solution', linewidth=2)
plt.plot(nodes, u_fem, 'bo-', label='FEM solution', markersize=6, linewidth=1.5)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('1D Poisson Equation: $-u_{xx} = \sin(x)$ with Random Node Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Error analysis
u_exact_nodes = u_exact(nodes)
error = np.abs(u_fem - u_exact_nodes)

max_error = np.max(error)
rms_error = np.sqrt(np.mean(error**2))
mean_error = np.mean(error)
rel_error = rms_error / np.max(np.abs(u_exact_nodes))

print("\n=== ERROR ANALYSIS ===")
print("Function: f(x) = sin(x)")
print("Exact solution: u(x) = sin(x) - x·sin(1)")
print(f"Maximum error      : {max_error:.2e}")
print(f"Mean absolute error: {mean_error:.2e}")
print(f"RMS error          : {rms_error:.2e}")
print(f"Relative RMS error : {rel_error:.2e}")

print("\n=== FEM vs Exact Solution per Node ===")
print(f"{'Node':>4} | {'x':>8} | {'u_FEM':>12} | {'u_exact':>12} | {'|Error|':>10}")
print("-" * 55)

for i in range(len(nodes)):
    err = abs(u_fem[i] - u_exact(nodes[i]))
    print(f"{i:4d} | {nodes[i]:8.4f} | {u_fem[i]:12.6f} | {u_exact(nodes[i]):12.6f} | {err:10.2e}")
print("\nInterpretation:")
if rms_error < 1e-3:
    print(" → The FEM solution is extremely accurate — errors are within numerical tolerance.")
elif rms_error < 1e-2:
    print(" → The FEM captures the main shape of the exact solution with minor deviations.")
else:
    print(" → The discretization may be too coarse or the integration rule too simple.")
