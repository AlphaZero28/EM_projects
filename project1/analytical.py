"""
Analytical (ideal) amplifier output of the field mill.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 18

# Parameters
eps0 = 8.8541878128e-12
E0 = 100.0  # V/m
Rf = 1e6  # ohm
v = 1e-3  # m/s
w = 50e-3  # plate width perpendicular to the motion (m)
L = 15e-3  # plate length along the motion (m)
gap = 2e-3  # gap between the plates (m)

# Shutter position
s = np.linspace(0, L + gap, 1000)

# Exposed area of each plate
# Plate 1 is uncovered as the shutter leaves it (s from 0 to L)
a1 = w * np.clip(s, 0, L)
# Plate 2 is covered once the shutter's right edge passes it 
a2 = w * (L - np.clip(s - gap, 0, L))

# Induced charge on each plate
q1 = -eps0 * E0 * a1
q2 = -eps0 * E0 * a2

# Current out of each plate into the amplifier
i1 = -v * np.gradient(q1, s)
i2 = -v * np.gradient(q2, s)

# Differential transimpedance amplifier
V_out = -Rf * (i1 - i2)


# Plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(s * 1e3, V_out * 1e9)
ax.set_xlabel("Shutter position (mm)")
ax.set_ylabel("Output voltage (nV)")
fig.tight_layout()
fig.savefig("analytical.png", dpi=150, bbox_inches="tight")
plt.show()

# np.savez("analytical_data.npz", s=s, V_out=V_out)
