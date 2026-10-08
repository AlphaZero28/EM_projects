
import time
import numpy as np
import matplotlib.pyplot as plt
from functions import Geometry

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 18

E0 = 100.0
w_out = 50e-3


def run_case(domain_width, domain_height, h):
    """Solve one case and return (Q1, Q2) in coulombs."""
    plate_length, plate_thickness, plate_height, plate_gap = 15e-3, 0.5e-3, 10e-3, 2e-3
    yp0 = plate_height
    yp1 = yp0 + plate_thickness
    pair_length = 2 * plate_length + plate_gap
    x0 = round(0.5 * (domain_width - pair_length) / h) * h
    ys0 = yp1 + 2e-3
    ys1 = ys0 + 0.5e-3 

    x = np.arange(round(domain_width / h) + 1) * h
    y = np.arange(round(domain_height / h) + 1) * h
    X, Y = np.meshgrid(x, y)

    g = Geometry(X, Y)
    g.min_y_bc(0.0)
    g.max_y_bc(E0 * domain_height)
    g.min_x_bc(0.0)
    g.max_x_bc(0.0)
    g.add_conductor("Plate 1", x0, x0 + plate_length, yp0, yp1)
    g.add_conductor("Plate 2", x0 + plate_length + plate_gap, x0 + pair_length, yp0, yp1)
    g.add_conductor("Shutter", x0, x0 + plate_length, ys0, ys1)
    g.number_unknowns()
    phi = g.solve()
    return (g.charge(phi, "Plate 1") * w_out,
            g.charge(phi, "Plate 2") * w_out)


# ---- Study 1: domain size (square box), grid fixed at 0.5 mm ----
sizes = [40, 50, 60, 80, 100, 120, 160]
domain = []
for size in sizes:
    t0 = time.time()
    domain.append(run_case(size * 1e-3, size * 1e-3, 0.5e-3))
    print(f"domain {size:4d} x {size:<4d} mm   Q1 = {domain[-1][0]:.4e} C   Q2 = {domain[-1][1]:.4e} C   ({time.time() - t0:.1f} s)")
domain = np.array(domain)

# ---- Study 2: grid spacing, domain fixed at 80 x 80 mm ----
steps = [1.0, 0.5, 0.25, 0.125]
grid = []
for h in steps:
    t0 = time.time()
    grid.append(run_case(80e-3, 80e-3, h * 1e-3))
    print(f"grid {h:5.3f} mm         Q1 = {grid[-1][0]:.4e} C   Q2 = {grid[-1][1]:.4e} C   ({time.time() - t0:.1f} s)")
grid = np.array(grid)

# ---- Change in the calculated charge between one case and the previous one ----
def change(Q):
    """Percent change of each case relative to the case before it."""
    return 100 * np.abs((Q[1:] - Q[:-1]) / Q[1:])

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(sizes[1:], change(domain[:, 0]), "o-", label="Plate 1")
ax1.plot(sizes[1:], change(domain[:, 1]), "o-", label="Plate 2")
ax1.set_xlabel("Domain size (mm)")
ax1.set_ylabel("Change in Q from previous case (%)")
ax1.set_title("step size = 0.5 mm")
ax1.legend()

ax2.plot(steps[1:], change(grid[:, 0]), "o-", label="Plate 1")
ax2.plot(steps[1:], change(grid[:, 1]), "o-", label="Plate 2")
ax2.set_xlabel("Grid spacing (mm)")
ax2.set_title("Domain 80 x 80 mm")
ax2.legend()

fig.tight_layout()
fig.savefig("convergence.png", dpi=150, bbox_inches="tight")
plt.show()
