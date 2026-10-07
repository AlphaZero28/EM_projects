"""
* author: Ohidul Islam
* copyright 2026
"""

import numpy as np
from functions import Geometry, plot_geometry


def snap(value, step):
    """Round a length to the nearest grid line."""
    return round(value / step) * step


# Grid spacing
dx = 0.25e-3  # 0.25 mm
dy = 0.25e-3  # 0.25 mm

# Physical and electrical parameters
E0 = 100.0  # V/m
w_out = 50e-3  # plate depth W 
Rf = 1e6  # ohm
v = 1e-3  # m/s

# Sensing plates
plate_length = 15e-3
plate_thickness = 0.5e-3
plate_height = 10e-3  # bottom of plates above y = 0
plate_gap = 2e-3

yp0 = plate_height
yp1 = yp0 + plate_thickness
pair_length = 2 * plate_length + plate_gap

# Domain size 
side_margin_in_L = 3.5
height_in_L = 10.0

x_pair_left = snap(side_margin_in_L * plate_length, dx)
domain_width = snap(2 * x_pair_left + pair_length, dx)
domain_height = snap(height_in_L * plate_length, dy)

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
number_of_x_points = round(domain_width / dx) + 1
number_of_y_points = round(domain_height / dy) + 1
x = np.arange(number_of_x_points) * dx
y = np.arange(number_of_y_points) * dy
X, Y = np.meshgrid(x, y)

top_voltage = E0 * domain_height


def build_geometry(n):
    s = s_start + n * dx

    geometry = Geometry(X, Y)
    geometry.add_rectangle("Plate 1", plate1_x0, plate1_x1, yp0, yp1)
    geometry.add_rectangle("Plate 2", plate2_x0, plate2_x1, yp0, yp1)
    geometry.add_rectangle("Shutter", s, s + shutter_length, ys0, ys1)
    geometry.assign_materials(X, Y)

    geometry.min_y_bc(0.0)  # base
    geometry.max_y_bc(top_voltage)  # top
    geometry.min_x_bc(0.0)  # left wall: dV/dx = 0
    geometry.max_x_bc(0.0)  # right wall: dV/dx = 0

    geometry.set_potential("Plate 1", 0.0)
    geometry.set_potential("Plate 2", 0.0)
    geometry.set_potential("Shutter", 0.0)

    geometry.finalize()
    return geometry


if __name__ == "__main__":
    print(f"Domain: {domain_width * 1e3:.1f} mm x {domain_height * 1e3:.1f} mm")
    print(f"Grid:   {number_of_x_points} x {number_of_y_points} nodes")
    print(f"Shutter positions: {Ns}")

    first = build_geometry(0)
    plot_geometry(first, x, y, top_voltage=top_voltage)