"""
Amplifier output from the sweep data (sweep_data.npz).
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 18

# Parameters
Rf = 1e6  # ohm
v = 1e-3  # m/s

# Load the charge on each plate at every shutter position
data = np.load("sweep_data.npz")
s = data["s"] - data["s"][0]   # shutter position, 0 = covering Plate 1 (m)
Q1 = data["Q1"]
Q2 = data["Q2"]

# Current flowing out of each plate into the amplifier: i_in = -dQ/dt = -v * dQ/ds
i1 = -v * np.gradient(Q1, s)
i2 = -v * np.gradient(Q2, s)

# Differential connection, then the transimpedance amplifier
i = i1 - i2
V_out = -i * Rf

print(f"Output in the middle of the stroke: {V_out[len(s) // 2] * 1e9:.1f} nV")
print(f"Largest output:                     {np.abs(V_out).max() * 1e9:.1f} nV")

# Plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(s * 1e3, V_out * 1e9)
ax.set_xlabel("Shutter position (mm)")
ax.set_ylabel("Output voltage (nV)")
fig.tight_layout()
fig.savefig("output.png", dpi=150, bbox_inches="tight")
plt.show()

np.savez("output_data.npz", s=s, V_out=V_out)
