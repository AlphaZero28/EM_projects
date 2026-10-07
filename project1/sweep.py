"""
 * author Ohidul Islam
 * copyright 2026
"""

"""
Shutter sweep: solve the field for every shutter position and record the
charge on Plate 1 and Plate 2.  Results are saved to sweep_data.npz.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from functions import Geometry

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 18

# Grid spacing
dx = 0.25e-3  # 0.25 mm
dy = 0.25e-3  # 0.25 mm

# Physical and electrical parameters
E0 = 100.0  # V/m
w_out = 50e-3  # plate depth W (out of plane)

# Domain size
domain_width = 80e-3  # 80 mm
domain_height = 80e-3  # 80 mm

# Sensing plates
plate_length = 15e-3
plate_thickness = 0.5e-3
plate_height = 10e-3  # bottom of plates above y = 0
plate_gap = 2e-3

yp0 = plate_height
yp1 = yp0 + plate_thickness
pair_length = 2 * plate_length + plate_gap

# Centre the plate pair in x, on a grid line
x_pair_left = round(0.5 * (domain_width - pair_length) / dx) * dx

plate1_x0 = x_pair_left
plate1_x1 = plate1_x0 + plate_length
plate2_x0 = plate1_x1 + plate_gap
plate2_x1 = plate2_x0 + plate_length

# Shutter (same length as a plate)
shutter_length = plate_length
shutter_thickness = 0.5e-3
shutter_plate_gap = 2e-3

ys0 = yp1 + shutter_plate_gap
ys1 = ys0 + shutter_thickness

# Shutter positions, one grid cell per step
s_start = plate1_x0
s_end = plate2_x1 - shutter_length
Ns = round((s_end - s_start) / dx) + 1

# Grid
x = np.arange(round(domain_width / dx) + 1) * dx
y = np.arange(round(domain_height / dy) + 1) * dy
X, Y = np.meshgrid(x, y)

# Top boundary voltage gives a far-field of E0
top_voltage = E0 * domain_height

# ---- Sweep ----
s_values = s_start + np.arange(Ns) * dx   # left edge of the shutter
Q1 = np.zeros(Ns)
Q2 = np.zeros(Ns)

for k, s in enumerate(s_values):
    t0 = time.time()

    geometry = Geometry(X, Y)
    geometry.min_y_bc(0.0)
    geometry.max_y_bc(top_voltage)
    geometry.min_x_bc(0.0)
    geometry.max_x_bc(0.0)
    geometry.add_conductor("Plate 1", plate1_x0, plate1_x1, yp0, yp1)
    geometry.add_conductor("Plate 2", plate2_x0, plate2_x1, yp0, yp1)
    geometry.add_conductor("Shutter", s, s + shutter_length, ys0, ys1)
    geometry.number_unknowns()
    phi = geometry.solve()

    Q1[k] = geometry.charge(phi, "Plate 1") * w_out
    Q2[k] = geometry.charge(phi, "Plate 2") * w_out
    print(f"{k + 1:3d}/{Ns}  s = {(s - s_start) * 1e3:5.2f} mm   "
          f"Q1 = {Q1[k]:.4e} C   Q2 = {Q2[k]:.4e} C   ({time.time() - t0:.1f} s)")

# Save so we never have to solve again
np.savez("sweep_data.npz", s=s_values, Q1=Q1, Q2=Q2)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot((s_values - s_start) * 1e3, Q1, label="Plate 1")
ax.plot((s_values - s_start) * 1e3, Q2, label="Plate 2")
ax.set_xlabel("Shutter position (mm)")
ax.set_ylabel("Charge (C)")
ax.legend()
fig.tight_layout()
fig.savefig("sweep.png", dpi=150, bbox_inches="tight")
plt.show()