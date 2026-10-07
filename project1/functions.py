import numpy as np


class Geometry:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dx = X[0, 1] - X[0, 0]
        self.dy = Y[1, 0] - Y[0, 0]

        # The one array that matters: NaN = unknown, number = fixed voltage.
        self.potential = np.full(X.shape, np.nan)

        # d(potential)/dx on the left and right walls.
        self.min_x_gradient = 0.0
        self.max_x_gradient = 0.0

        # Kept only so plot_geometry can draw the objects.
        self.rectangles = []

    # ---- Boundary conditions ----
    def min_y_bc(self, voltage):
        """Potential on the bottom boundary."""
        self.potential[0, :] = voltage

    def max_y_bc(self, voltage):
        """Potential on the top boundary."""
        self.potential[-1, :] = voltage

    def min_x_bc(self, gradient):
        """d(potential)/dx on the left boundary."""
        self.min_x_gradient = gradient

    def max_x_bc(self, gradient):
        """d(potential)/dx on the right boundary."""
        self.max_x_gradient = gradient

    # ---- Conductors ----
    def add_conductor(self, name, x_left, x_right, y_bottom, y_top, voltage=0.0):
        """Fix the potential of every node inside the rectangle."""
        tol = 1e-6 * min(self.dx, self.dy)
        inside = ((self.X >= x_left - tol) & (self.X <= x_right + tol)
                  & (self.Y >= y_bottom - tol) & (self.Y <= y_top + tol))
        self.potential[inside] = voltage

        self.rectangles.append({
            "name": name,
            "x_left": x_left,
            "y_bottom": y_bottom,
            "width": x_right - x_left,
            "height": y_top - y_bottom,
        })

    # ---- Node classification ----
    @property
    def unknown(self):
        return np.isnan(self.potential)

    @property
    def fixed(self):
        return ~np.isnan(self.potential)

    # ---- Plots ----
    def plot_geometry(self):
        """Draw the domain, the conductors and the boundary conditions."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        x_max = self.X[0, -1] * 1000
        y_max = self.Y[-1, 0] * 1000
        top_voltage = self.potential[-1, 0]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        for r in self.rectangles:
            ax.add_patch(Rectangle(
                (r["x_left"] * 1000, r["y_bottom"] * 1000),
                r["width"] * 1000, r["height"] * 1000,
                facecolor="lightgreen", edgecolor="black",
            ))

        plt.show()

    def plot_potential(self, phi=None, xlim=None, ylim=None, filename=None):
        """Plot a potential array. NaN (unknown) nodes are drawn in grey.

        With no argument it plots self.potential (only fixed nodes are
        coloured). Later, pass the solved array as phi.
        xlim, ylim are (min, max) in millimetres, to zoom in.
        """
        import matplotlib.pyplot as plt

        if phi is None:
            phi = self.potential

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgrey")

        fig, ax = plt.subplots(figsize=(9, 6))
        mesh = ax.pcolormesh(
            self.X[0, :] * 1000, self.Y[:, 0] * 1000,
            np.ma.masked_invalid(phi), shading="nearest", cmap=cmap,
        )
        fig.colorbar(mesh, ax=ax)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Potential (V)")

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()