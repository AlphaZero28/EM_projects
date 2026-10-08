
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 18

# Parameters
eps0 = 8.8541878128e-12
E0 = 100.0  
Rf = 1e6  
v = 1e-3  
w = 50e-3  # plate width 
L = 15e-3  # plate length 
gap = 2e-3  # gap between the plates

# Shutter position
s = np.linspace(0, L + gap, 1000)

# Exposed area 
a1 = w * np.clip(s, 0, L)
a2 = w * (L - np.clip(s - gap, 0, L))

# Induced charge 
q1 = -eps0 * E0 * a1
q2 = -eps0 * E0 * a2

# Current out of each plate into the amplifier
i1 = -v * np.gradient(q1, s)
i2 = -v * np.gradient(q2, s)

# Differential transimpedance amplifier
V_out = Rf * (i1 - i2)


# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(s * 1e3, V_out * 1e9, lw=2)
ax.set_xlabel("Shutter position (mm)")
ax.set_ylabel("Output voltage (nV)")
fig.tight_layout()
fig.savefig("analytical.png", dpi=150, bbox_inches="tight")
plt.show()

# np.savez("analytical_data.npz", s=s, V_out=V_out)
