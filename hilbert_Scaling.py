import numpy as np
import matplotlib.pyplot as plt

# Range of system sizes
N = np.arange(2, 26)  # up to 25 spins
dim = 2**N

plt.figure(figsize=(6, 4))
plt.semilogy(N, dim, "o-", linewidth=2, markersize=6)

# Highlight key points
for n in [10, 20, 24]:
    plt.semilogy(n, 2**n, "ro")
    plt.text(n, 2**n * 1.2, f"N={n}", ha="center", fontsize=9)

plt.xlabel("Number of spins (N)", fontsize=12)
plt.ylabel("Hilbert space dimension (2^N)", fontsize=12)
plt.title("Exponential growth of Hilbert space dimension", fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

plt.tight_layout()
plt.savefig("hilbert_scaling.png", dpi=300)
plt.show()
