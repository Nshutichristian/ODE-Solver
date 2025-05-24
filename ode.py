"""
CST-305 Project 3: Green's Function and Homogeneous ODE Solver

Programmer: Christian Nshuti Manzi

Packages Used:
- numpy: for numerical evaluation
- matplotlib: for plotting graphs
- sympy: for symbolic math and solving ODEs

Purpose:
- Solve two ODEs using homogeneous and Green's Function approaches.
- Display and save four plots:
    1. homogeneous_ode1.png
    2. homogeneous_ode2.png
    3. greens_ode1.png
    4. greens_ode2.png
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, cos, sin, simplify, integrate, lambdify

# Define variables and functions
t, x = symbols('t x')
y = Function('y')

# === Define and Solve ODEs ===

# ODE 1: y'' + y = 4 → homogeneous: y'' + y = 0
ode1 = Eq(y(t).diff(t, 2) + y(t), 4)
hom_ode1 = Eq(y(t).diff(t, 2) + y(t), 0)

# ODE 2: y'' + 4y = x → homogeneous: y'' + 4y = 0
ode2 = Eq(y(x).diff(x, 2) + 4 * y(x), x)
hom_ode2 = Eq(y(x).diff(x, 2) + 4 * y(x), 0)

# Solve homogeneous ODEs (without ICs to get general solution)
hom_gen_sol1 = dsolve(hom_ode1, y(t))
hom_gen_sol2 = dsolve(hom_ode2, y(x))

# Green's Function solutions (known results)
greens_sol1_expr = 4 - 4 * cos(t)
greens_sol2_expr = x / 4 - sin(2 * x) / 8

print("ODE 1: y'' + y = 4 with y(0)=0, y'(0)=0")
print("ODE 2: y'' + 4y = x with y(0)=0, y'(0)=0\n")

print("General Homogeneous Solution ODE 1:", hom_gen_sol1.rhs)
print("General Homogeneous Solution ODE 2:", hom_gen_sol2.rhs)
print("Green's Function Solution ODE 1: y(t) =", greens_sol1_expr)
print("Green's Function Solution ODE 2: y(x) =", greens_sol2_expr)

# === Numerical Plotting Setup ===
t_vals = np.linspace(0, 10, 400)
x_vals = np.linspace(0, 10, 400)

# For homogeneous solutions, we'll plot the general solution with arbitrary constants
# Let's choose C1=1, C2=0 for visualization purposes
C1, C2 = symbols('C1 C2')
hom_sol1 = hom_gen_sol1.rhs.subs({C1: 1, C2: 0})  # y(t) = cos(t)
hom_sol2 = hom_gen_sol2.rhs.subs({C1: 1, C2: 0})  # y(x) = cos(2x)

# Homogeneous Solution ODE 1
hom_func1 = lambdify(t, hom_sol1, modules='numpy')
y_vals1 = hom_func1(t_vals)
plt.figure()
plt.plot(t_vals, y_vals1, label="Homogeneous: $y(t) = \cos(t)$", color="blue")
plt.title("Homogeneous Solution: $y'' + y = 0$")
plt.xlabel("Time t [s]")
plt.ylabel("Displacement y(t)")
plt.grid(True)
plt.legend()
plt.savefig("homogeneous_ode1.png")
plt.close()  # Close the figure to free memory

# Homogeneous Solution ODE 2
hom_func2 = lambdify(x, hom_sol2, modules='numpy')
y_vals2 = hom_func2(x_vals)
plt.figure()
plt.plot(x_vals, y_vals2, label="Homogeneous: $y(x) = \cos(2x)$", color="orange")
plt.title("Homogeneous Solution: $y'' + 4y = 0$")
plt.xlabel("Position x [m]")
plt.ylabel("Amplitude y(x)")
plt.grid(True)
plt.legend()
plt.savefig("homogeneous_ode2.png")
plt.close()

# Green's Function ODE 1
greens_func1 = lambdify(t, greens_sol1_expr, modules='numpy')
y_vals3 = greens_func1(t_vals)
plt.figure()
plt.plot(t_vals, y_vals3, label="Green's: $y(t) = 4 - 4\cos(t)$", color="green")
plt.title("Green's Function Solution: $y'' + y = 4$")
plt.xlabel("Time t [s]")
plt.ylabel("Displacement y(t)")
plt.grid(True)
plt.legend()
plt.savefig("greens_ode1.png")
plt.close()

# Green's Function ODE 2
greens_func2 = lambdify(x, greens_sol2_expr, modules='numpy')
y_vals4 = greens_func2(x_vals)
plt.figure()
plt.plot(x_vals, y_vals4, label="Green's: $y(x) = \\frac{x}{4} - \\frac{\sin(2x)}{8}$", color="purple")
plt.title("Green's Function Solution: $y'' + 4y = x$")
plt.xlabel("Position x [m]")
plt.ylabel("Amplitude y(x)")
plt.grid(True)
plt.legend()
plt.savefig("greens_ode2.png")
plt.close()

# === Output Confirmation ===
print("\n✅ All four plots generated and saved:")
print(" - homogeneous_ode1.png")
print(" - homogeneous_ode2.png")
print(" - greens_ode1.png")
print(" - greens_ode2.png")