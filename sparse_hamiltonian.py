import matplotlib.pyplot as plt
import numpy as np

# Parameters for a toy sparse Hamiltonian example
N = 16  # matrix dimension
matrix = np.zeros((N, N))

# Add a block-diagonal sparse pattern (toy-Ising-like structure)
for i in range(N - 1):
    matrix[i, i] = np.random.choice([-1, 1])  # diagonal terms
    matrix[i, i + 1] = -1  # nearest-neighbour coupling
    matrix[i + 1, i] = -1

matrix[N - 1, N - 1] = np.random.choice([-1, 1])

# Plot sparsity pattern
plt.figure(figsize=(5, 5))
plt.spy(matrix, markersize=6, color="black")
plt.title("Sparse Hamiltonian Structure (Toy Example)", fontsize=12)
plt.xlabel("Column Index")
plt.ylabel("Row Index")

plt.tight_layout()
plt.savefig("sparse_hamiltonian.png", dpi=300)
plt.show()
# plt.close()
