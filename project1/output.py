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

# Load the charge on each plate 
data = np.load("data/sweep_data.npz")
s = data["s"] - data["s"][0]   
Q1 = data["Q1"]
Q2 = data["Q2"]

# Current 
i1 = -v * np.gradient(Q1, s)
i2 = -v * np.gradient(Q2, s)

# Differential connection
i = i1 - i2
V_out = i * Rf


# Plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(s * 1e3, V_out * 1e9, lw=2)
ax.set_xlabel("Shutter position (mm)")
ax.set_ylabel("Output voltage (nV)")
fig.tight_layout()
fig.savefig("output.png", dpi=150, bbox_inches="tight")
plt.show()

# np.savez("data/output_data.npz", s=s, V_out=V_out)
