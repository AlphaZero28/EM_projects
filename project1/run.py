"""
* author: Ohidul Islam
* created on 06-10-2026-14h-17m
* copyright 2026
"""

import numpy as np
from functions import Geometry
import matplotlib.pyplot as plt


# Grid spacing
dx = 0.25e-3  # 0.25 mm
dy = 0.25e-3  # 0.25 mm

# Physical and electrical parameters
E0 = 100.0  # V/m
w_out = 50e-3  # plate depth W (out of plane)
Rf = 1e6  # ohm
v = 1e-3  # m/s

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

# Shutter position used now: the first one (covers Plate 1)
s = s_start

# Grid
number_of_x_points = round(domain_width / dx) + 1
number_of_y_points = round(domain_height / dy) + 1
x = np.arange(number_of_x_points) * dx
y = np.arange(number_of_y_points) * dy
X, Y = np.meshgrid(x, y)

# Top boundary voltage gives a far-field of E0
top_voltage = E0 * domain_height

# Build the geometry
geometry = Geometry(X, Y)

geometry.min_y_bc(0.0)  # base
geometry.max_y_bc(top_voltage)  # top
geometry.min_x_bc(0.0)  # left wall: dV/dx = 0
geometry.max_x_bc(0.0)  # right wall: dV/dx = 0
geometry.plot_potential(filename="potential_initial.png")
# Conductors: every node inside is fixed at 0 V
geometry.add_conductor("Plate 1", plate1_x0, plate1_x1, yp0, yp1)
geometry.add_conductor("Plate 2", plate2_x0, plate2_x1, yp0, yp1)
geometry.add_conductor("Shutter", s, s + shutter_length, ys0, ys1)

# Summary
print(f"Domain: {domain_width * 1e3:.1f} mm x {domain_height * 1e3:.1f} mm")
print(f"Grid:   {number_of_x_points} x {number_of_y_points} nodes")
print(f"Shutter positions: {Ns}")
print(f"Fixed nodes:   {geometry.fixed.sum()}")
print(f"Unknown nodes: {geometry.unknown.sum()}")

# Number the unknown nodes (one equation per unknown node)
geometry.number_unknowns()
print(f"Largest equation number + 1: {geometry.index.max() + 1}")
print(f"Fixed nodes all -1: {(geometry.index[geometry.fixed] == -1).all()}")

# Neighbours (left, right, below, above) of two nodes, as a check
print("Node [22, 30] (just above Plate 1):", geometry.neighbors(22, 30))
print("Node [50, 0]  (left wall):         ", geometry.neighbors(50, 0))

# Equation rows for two nodes, as a check
print("Equation for node [22, 30] (just above Plate 1):", geometry.equation_row(22, 30))
print("Equation for node [50, 0]  (left wall):         ", geometry.equation_row(50, 0))

# Plots
geometry.plot_geometry()

# Potential array: grey = unknown, coloured = fixed. Zoomed on the plates.
geometry.plot_potential(filename="potential_with_geo.png")
phi = geometry.solve()
geometry.plot_potential(phi, filename="potential_solved.png")

# Electric field from the solved potential, zoomed on the plates
Ex, Ey = geometry.electric_field(phi)
geometry.plot_field(phi, xlim=(0, 50), ylim=(0, 25), filename="field.png")

# Check: how many nodes does each conductor own?
for name, mask in geometry.masks.items():
    print(name, mask.sum(), "nodes")

# Charge on each conductor (per unit depth, times the plate depth w_out)
eps0 = 8.8541878128e-12
Q_ideal = eps0 * E0 * w_out * plate_length
for name in geometry.masks:
    Q = geometry.charge(phi, name) * w_out
    print(f"{name}: Q = {Q:.3e} C   ({Q / Q_ideal:.2f} x ideal)")

# Neutrality check: all charges together must add up to zero
Q_base = geometry.boundary_charge(phi, 0) * w_out
Q_top = geometry.boundary_charge(phi, -1) * w_out
Q_all = sum(geometry.charge(phi, name) for name in geometry.masks) * w_out
print(f"Base: {Q_base:.3e} C   Top: {Q_top:.3e} C")
print(f"Sum of everything: {Q_all + Q_base + Q_top:.3e} C")
