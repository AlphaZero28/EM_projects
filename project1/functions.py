import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 18
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

        # One True/False array per conductor, same shape as potential.
        self.masks = {}

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
        self.masks[name] = inside              # remember which nodes are this conductor

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

    # ---- Equation numbering ----
    def number_unknowns(self):
        """Give every unknown node an equation number 0, 1, 2, ...

        """
        self.index = np.full(self.potential.shape, -1, dtype=int)
        self.index[self.unknown] = np.arange(self.unknown.sum())

    # ---- Neighbours ----
    def neighbors(self, j, i):
        """Equation numbers of the four neighbours of node [j, i].

        Returns (left, right, below, above). For each neighbour:
          >= 0  -> unknown node, this is its equation number
          -1    -> fixed node (its voltage is in self.potential)
          None  -> no such node (off the grid, e.g. beyond a side wall)
        """
        Ny, Nx = self.index.shape
        result = []
        for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            jn, i_n = j + dj, i + di
            if 0 <= jn < Ny and 0 <= i_n < Nx:
                result.append(int(self.index[jn, i_n]))
            else:
                result.append(None)
        return tuple(result)

    # ---- Equation row ----
    def equation_row(self, j, i):
        """Row of the equation for unknown node [j, i].

        Returns (columns, coefficients, rhs):
          columns      -> equation numbers of the unknowns in this equation
          coefficients -> the number that multiplies each of those unknowns
          rhs          -> right-hand side, from fixed neighbours and wall slopes
        """
        k = int(self.index[j, i])
        if k < 0:
            raise ValueError(f"Node [{j}, {i}] is fixed, it has no equation.")

        left, right, below, above = self.neighbors(j, i)
        positions = [(j, i - 1), (j, i + 1), (j - 1, i), (j + 1, i)]
        rhs = 0.0

        # Side walls: the missing neighbour is a ghost node that mirrors the
        # neighbour on the other side, corrected by the wall slope.
        if left is None:       # V_left = V_right - 2*dx*slope
            left, positions[0] = right, positions[1]
            rhs += -0.5 * self.dx * self.min_x_gradient
        if right is None:      # V_right = V_left + 2*dx*slope
            right, positions[1] = left, positions[0]
            rhs += 0.5 * self.dx * self.max_x_gradient

        terms = {k: 1.0}                       # the node itself
        for neighbor, (jn, i_n) in zip((left, right, below, above), positions):
            if neighbor is None:
                raise ValueError(f"Node [{j}, {i}] is on the top or bottom edge.")
            if neighbor >= 0:                  # unknown neighbour
                terms[neighbor] = terms.get(neighbor, 0.0) - 0.25
            else:                              # fixed neighbour
                rhs += 0.25 * float(self.potential[jn, i_n])

        return list(terms), list(terms.values()), float(rhs)


    # ---- Build the system A · V = b ----
    def build_system(self):
        """Build the sparse matrix A and the vector b for A · V = b."""
        n = int(self.unknown.sum())            # number of equations = unknowns
        rows, cols, vals = [], [], []
        b = np.zeros(n)

        for j, i in zip(*np.nonzero(self.unknown)):
            k = int(self.index[j, i])          # this node's row number
            columns, coefficients, rhs = self.equation_row(j, i)
            for column, coefficient in zip(columns, coefficients):
                rows.append(k)
                cols.append(column)
                vals.append(coefficient)
            b[k] = rhs

        A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        return A, b
    
    # ---- Solve ----
    def solve(self):
        """Solve A · V = b and return the full potential map."""
        A, b = self.build_system()
        V = spsolve(A, b)                  # potential at every unknown node
        phi = self.potential.copy()        # fixed nodes already have their voltage
        phi[self.unknown] = V              # put the unknowns back on the map
        return phi        
    def boundary_charge(self, phi, row):
        """Charge per unit depth (C/m) on the base (row=0) or top (row=-1) boundary.

        Same link sum as charge(), but the end nodes (on the side walls)
        only own half a cell, so they count half.
        """
        eps0 = 8.8541878128e-12
        inside = 1 if row == 0 else -2          # the row of nodes just inside the domain
        weights = np.ones(phi.shape[1])
        weights[0] = weights[-1] = 0.5
        return eps0 * np.sum(weights * (phi[row, :] - phi[inside, :]))

    # ---- Electric field ----
    def electric_field(self, phi):
        """E = -grad(V). Returns Ex, Ey in V/m (NaN inside the conductors)."""
        dV_dy, dV_dx = np.gradient(phi, self.dy, self.dx)
        Ex, Ey = -dV_dx, -dV_dy

        conductor = self.fixed.copy()
        conductor[0, :] = False        # base and top rows are boundaries,
        conductor[-1, :] = False       # not conductors
        Ex[conductor] = np.nan
        Ey[conductor] = np.nan
        return Ex, Ey

    # ---- Charge ----
    def charge(self, phi, name):
        """Charge per unit depth (C/m) on the conductor `name`.

        Square grid (dx = dy). Q/W = eps0 * sum of (V_conductor - V_neighbour)
        over every link from a conductor node to a node outside it.
        """
        eps0 = 8.8541878128e-12
        mask = self.masks[name]
        total = 0.0
        for j, i in zip(*np.nonzero(mask)):
            for jn, i_n in ((j, i - 1), (j, i + 1), (j - 1, i), (j + 1, i)):
                if not mask[jn, i_n]:                  # neighbour is outside the conductor
                    total += phi[j, i] - phi[jn, i_n]
        return eps0 * total

    # ---- Plots ----
    def plot_geometry(self):
        """Draw the domain, the conductors and the boundary conditions."""


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

    def plot_field(self, phi, xlim=None, ylim=None, filename=None):
        """Plot |E| as colour, with field lines. xlim, ylim in millimetres."""
        Ex, Ey = self.electric_field(phi)
        magnitude = np.hypot(Ex, Ey)
        E0 = self.potential[-1, 0] / self.Y[-1, 0]   # applied field, V/m

        x = self.X[0, :] * 1000
        y = self.Y[:, 0] * 1000

        fig, ax = plt.subplots(figsize=(9, 6))
        mesh = ax.pcolormesh(
            x, y, np.ma.masked_invalid(magnitude), shading="nearest",
            cmap="inferno", vmin=0, vmax=4 * E0,
        )
        fig.colorbar(mesh, ax=ax, label="|E| (V/m)")

        ax.streamplot(
            x, y, np.ma.masked_invalid(Ex), np.ma.masked_invalid(Ey),
            color="white", density=2, linewidth=0.5, arrowsize=0.7,
        )

        for r in self.rectangles:
            ax.add_patch(Rectangle(
                (r["x_left"] * 1000, r["y_bottom"] * 1000),
                r["width"] * 1000, r["height"] * 1000,
                facecolor="lightgreen", edgecolor="black",
            ))

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Electric field")

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()
