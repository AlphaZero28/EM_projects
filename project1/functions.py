import numpy as np


class Geometry:
    def __init__(self, X, Y):
        self.shape = X.shape
        self.dx = X[0, 1] - X[0, 0]
        self.dy = Y[1, 0] - Y[0, 0]

        self.rectangles = []

        # NaN means "unknown"; a number means "fixed voltage".
        self.potential = np.full(X.shape, np.nan)

        # d(potential)/dx on the left and right walls.
        self.min_x_gradient = 0.0
        self.max_x_gradient = 0.0

        self.conductor = np.zeros(X.shape, dtype=bool)

    # ---- Boundary conditions ----
    def min_x_bc(self, gradient):
        """d(potential)/dx on the left boundary."""
        self.min_x_gradient = gradient

    def max_x_bc(self, gradient):
        """d(potential)/dx on the right boundary."""
        self.max_x_gradient = gradient

    def min_y_bc(self, voltage):
        """Potential on the bottom boundary."""
        self.potential[0, :] = voltage

    def max_y_bc(self, voltage):
        """Potential on the top boundary."""
        self.potential[-1, :] = voltage

    # ---- Objects ----
    def add_rectangle(self, name, x_left, x_right, y_bottom, y_top,
                      material="conductor"):
        self.rectangles.append({
            "name": name,
            "x_left": x_left,
            "y_bottom": y_bottom,
            "width": x_right - x_left,
            "height": y_top - y_bottom,
            "material": material,
        })

    def assign_materials(self, X, Y):
        """Mark which grid points lie inside each rectangle."""
        self.conductor = np.zeros(X.shape, dtype=bool)
        tol = 1e-6 * min(self.dx, self.dy)

        for r in self.rectangles:
            x_left = r["x_left"]
            x_right = x_left + r["width"]
            y_bottom = r["y_bottom"]
            y_top = y_bottom + r["height"]

            inside = ((X >= x_left - tol) & (X <= x_right + tol)
                      & (Y >= y_bottom - tol) & (Y <= y_top + tol))
            r["points"] = inside
            if r["material"] == "conductor":
                self.conductor[inside] = True

    def set_potential(self, name, voltage):
        """Fix the voltage at every grid point inside the named rectangle."""
        for r in self.rectangles:
            if r["name"] == name:
                self.potential[r["points"]] = voltage
                return
        raise ValueError(f"Unknown rectangle: {name}")

    def finalize(self):
        """Sort nodes into fixed (known voltage) and unknown."""
        self.fixed = ~np.isnan(self.potential)
        if (self.conductor & ~self.fixed).any():
            raise ValueError("A conductor has no potential set.")
        self.unknown = ~self.fixed


def plot_geometry(geometry, x, y, top_voltage=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 7))
    x_max = x[-1] * 1000
    y_max = y[-1] * 1000
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    for r in geometry.rectangles:
        ax.add_patch(Rectangle(
            (r["x_left"] * 1000, r["y_bottom"] * 1000),
            r["width"] * 1000, r["height"] * 1000,
            facecolor="lightgreen", edgecolor="black",
        ))



    plt.show()