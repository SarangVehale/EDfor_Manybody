# Quick Start Guide - ED v2.0.0-corrected

## 30-Second Start

```bash
# 1. Validate (ALWAYS DO THIS FIRST!)
python exact_diagonalization_production.py --validate

# 2. Run your first calculation
python exact_diagonalization_production.py --model heisenberg --N 10 --Sz 0

# 3. Check results
cat ed_production_output/ed_production_summary.json
```

---

## Installation (2 minutes)

```bash
# Method 1: pip
pip install numpy scipy numba h5py

# Method 2: conda (recommended)
conda create -n ed python=3.10
conda activate ed
conda install numpy scipy numba h5py -c conda-forge

# Verify
python exact_diagonalization_production.py --validate
```

---

## Common Research Tasks

### Task 1: Ground State Energy

```bash
# Heisenberg chain
python exact_diagonalization_production.py \
    --model heisenberg \
    --N 12 \
    --Sz 0 \
    --J 1.0 \
    --output my_heisenberg

# Result in: ed_production_output/my_heisenberg_summary.json
```

### Task 2: Energy Gap

```bash
# Get ground state + first excited state
python exact_diagonalization_production.py \
    --model heisenberg \
    --N 12 \
    --Sz 0 \
    --k 10 \
    --output with_gap

# Gap = eigenvalues[1] - eigenvalues[0]
```

### Task 3: Parameter Scan

```bash
# Scan coupling strength
for J in 0.5 1.0 1.5 2.0; do
    python exact_diagonalization_production.py \
        --model heisenberg \
        --N 10 \
        --J $J \
        --output scan_J_${J}
done

# Collect results
grep "eigenvalues" ed_production_output/scan_J_*_summary.json
```

### Task 4: Phase Diagram Point

```bash
# TFIM at critical point (h = J)
python exact_diagonalization_production.py \
    --model tfim \
    --N 14 \
    --h 1.0 \
    --J 1.0 \
    --boundary periodic \
    --k 20 \
    --output tfim_critical
```

### Task 5: Hubbard Model

```bash
# Half-filling, strong coupling
python exact_diagonalization_production.py \
    --model hubbard \
    --N 8 \
    --t 1.0 \
    --U 8.0 \
    --N_up 4 \
    --N_dn 4 \
    --output hubbard_strong
```

---

## Reading Results

### Python

```python
import json
import h5py
import numpy as np

# Load JSON summary (quick)
with open('ed_production_output/my_calc_summary.json') as f:
    data = json.load(f)
    E0 = data['eigenvalues'][0]
    dim = data['dimensions']
    print(f"Ground state energy: {E0}")
    print(f"Hilbert space: {dim}")

# Load HDF5 (full data)
with h5py.File('ed_production_output/my_calc.h5', 'r') as f:
    eigenvalues = f['eigenvalues'][:]
    # eigenvectors = f['eigenvectors'][:]  # If saved
    N = f['metadata'].attrs['N']
```

### Command Line

```bash
# Quick look at eigenvalues
jq '.eigenvalues' ed_production_output/my_calc_summary.json

# Extract ground state energy
jq '.eigenvalues[0]' ed_production_output/my_calc_summary.json

# Check if Hermiticity passed
grep "Hermiticity validated" ed_production_output/my_calc_summary.json
```

---

## Troubleshooting (60 seconds)

### "Hermiticity check FAILED"

```
❌ Problem: Hamiltonian is not Hermitian
✅ Solution: This is a BUG! Report it immediately.
           Code should never produce non-Hermitian H.
```

### "MemoryError"

```
❌ Problem: System too large
✅ Solutions:
   1. Reduce N
   2. Use symmetry sectors (--Sz, --N_up, --N_dn)
   3. Use momentum sectors (--use_translation --momentum 0)
   4. Switch to DMRG for N > 18
```

### "Validation suite FAILED"

```
❌ Problem: Core physics is broken
✅ Solution: DO NOT USE for research!
           Contact maintainer or check for code corruption.
```

### "Results look wrong"

```
✅ Checklist:
   1. Did validation pass? (--validate)
   2. Is Hermiticity validated? (check logs)
   3. Are eigenvalue residuals small? (< 1e-8)
   4. Correct boundary conditions? (--boundary)
   5. Right quantum numbers? (--Sz, --N_up, --N_dn)
```

---

## System Size Limits (Know Before You Run)

| N   | Model      | Sector    | Dimension | RAM     | Time   | Device         |
| --- | ---------- | --------- | --------- | ------- | ------ | -------------- |
| 10  | Heisenberg | Sz=0      | 252       | < 10 MB | ~1s    | Laptop ✅      |
| 12  | Heisenberg | Sz=0      | 924       | ~50 MB  | ~10s   | Laptop ✅      |
| 14  | Heisenberg | Sz=0      | 3,432     | ~500 MB | ~2min  | Workstation ✅ |
| 16  | Heisenberg | Sz=0      | 12,870    | ~5 GB   | ~20min | Server ⚠️      |
| 18  | Heisenberg | Sz=0      | 48,620    | ~50 GB  | ~hours | HPC ⚠️         |
| 10  | TFIM       | Full      | 1,024     | ~10 MB  | ~1s    | Laptop ✅      |
| 14  | TFIM       | Full      | 16,384    | ~1 GB   | ~1min  | Laptop ✅      |
| 16  | TFIM       | Full      | 65,536    | ~10 GB  | ~10min | Workstation ⚠️ |
| 8   | Hubbard    | N↑=4,N↓=4 | ~100      | < 10 MB | ~1s    | Laptop ✅      |
| 10  | Hubbard    | N↑=5,N↓=5 | ~63,000   | ~1 GB   | ~5min  | Workstation ⚠️ |

**Rule of thumb**: If dim > 100,000, expect long waits or need HPC.

---

## Performance Tips (Get 10x Speedup)

### Use Symmetry Sectors

```bash
# Without sector (slow)
--model heisenberg --N 14
# dim = 16,384

# With Sz sector (4.8x faster!)
--model heisenberg --N 14 --Sz 0
# dim = 3,432
```

### Choose Right Method

```bash
# Small systems (N < 12): Use full diagonalization
--method full

# Medium (12 ≤ N ≤ 16): Use eigsh (default, best)
--method eigsh

# Large (N > 16): Use eigsh with small k
--method eigsh --k 6
```

### Parallel Parameter Scans

```bash
# Use GNU parallel for multi-point scans
parallel -j 4 python exact_diagonalization_production.py \
    --model heisenberg --N 10 --J {} --output J_{} \
    ::: 0.5 1.0 1.5 2.0 2.5 3.0
```

---

## Pre-Flight Checklist

Before submitting jobs or publishing results:

- [ ] ✅ Validation suite passed (`--validate`)
- [ ] ✅ Hermiticity check passed (see logs)
- [ ] ✅ Eigenvalue residuals < 1e-8 (see logs)
- [ ] ✅ Tested on small N first (quick check)
- [ ] ✅ Correct symmetry sectors chosen
- [ ] ✅ Boundary conditions correct
- [ ] ✅ Output files saved properly
- [ ] ✅ Results make physical sense

---

## Common Mistakes

### ❌ Mistake 1: Skip Validation

```bash
# WRONG: Jump straight to production
python exact_diagonalization_production.py --model hubbard --N 10

# RIGHT: Always validate first
python exact_diagonalization_production.py --validate
# Then proceed if validation passes
```

### ❌ Mistake 2: Wrong Quantum Numbers

```bash
# WRONG: Heisenberg with Sz > N/2 (impossible!)
--model heisenberg --N 10 --Sz 12  # ERROR: basis will be empty

# RIGHT: Sz must satisfy |Sz| ≤ N
--model heisenberg --N 10 --Sz 0  # Half-filling
--model heisenberg --N 10 --Sz 2  # 1 extra up spin
```

### ❌ Mistake 3: TFIM with Sz Sector

```bash
# WRONG: TFIM doesn't conserve Sz!
--model tfim --Sz 0  # This is ignored

# RIGHT: TFIM uses full basis
--model tfim --N 12  # Correct
```

### ❌ Mistake 4: Too Many Eigenvalues

```bash
# WRONG: k ≥ dim causes error
--model heisenberg --N 8 --Sz 0 --k 1000  # dim=70, k > dim!

# RIGHT: k < dim (code will auto-adjust)
--model heisenberg --N 8 --Sz 0 --k 10  # OK
```

---

## One-Line Examples (Copy-Paste Ready)

```bash
# Heisenberg AFM ground state
python exact_diagonalization_production.py --model heisenberg --N 12 --Sz 0 --J 1.0 --output heis_gs

# Heisenberg ferromagnetic (J < 0)
python exact_diagonalization_production.py --model heisenberg --N 10 --Sz 10 --J -1.0 --output heis_ferro

# TFIM quantum critical point
python exact_diagonalization_production.py --model tfim --N 14 --h 1.0 --J 1.0 --boundary periodic --output tfim_qcp

# TFIM in ordered phase (J >> h)
python exact_diagonalization_production.py --model tfim --N 12 --h 0.1 --J 1.0 --output tfim_ordered

# TFIM in paramagnetic phase (h >> J)
python exact_diagonalization_production.py --model tfim --N 12 --h 10.0 --J 1.0 --output tfim_para

# Hubbard half-filling, weak coupling
python exact_diagonalization_production.py --model hubbard --N 6 --t 1.0 --U 2.0 --N_up 3 --N_dn 3 --output hub_weak

# Hubbard half-filling, strong coupling (Mott)
python exact_diagonalization_production.py --model hubbard --N 6 --t 1.0 --U 8.0 --N_up 3 --N_dn 3 --output hub_mott

# Hubbard free fermions (U=0)
python exact_diagonalization_production.py --model hubbard --N 8 --t 1.0 --U 0.0 --N_up 4 --N_dn 4 --boundary periodic --output hub_free

# Hubbard atomic limit (t=0)
python exact_diagonalization_production.py --model hubbard --N 6 --t 0.0 --U 4.0 --N_up 3 --N_dn 3 --output hub_atomic

# High-precision calculation
python exact_diagonalization_production.py --model heisenberg --N 10 --Sz 0 --tol 1e-14 --k 20 --output high_precision

# Quick test (full diagonalization)
python exact_diagonalization_production.py --model heisenberg --N 8 --method full --output quick_test
```

---

## Quick Plotting (Optional)

### Extract and Plot Energy vs Parameter

```python
#!/usr/bin/env python3
import json
import glob
import matplotlib.pyplot as plt

# Collect data from parameter scan
J_values = []
E0_values = []

for filename in sorted(glob.glob('ed_production_output/scan_J_*_summary.json')):
    with open(filename) as f:
        data = json.load(f)
        J = data['config']['J']
        E0 = data['eigenvalues'][0]
        J_values.append(J)
        E0_values.append(E0)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(J_values, E0_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Coupling J', fontsize=14)
plt.ylabel('Ground State Energy E₀', fontsize=14)
plt.title('Heisenberg Chain Energy vs J', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('energy_vs_J.png', dpi=300)
plt.show()
```

### Plot Energy Gap (Phase Transition)

```python
import numpy as np
import matplotlib.pyplot as plt

# Assuming you did a scan over h for TFIM
h_values = []
gaps = []

for filename in sorted(glob.glob('ed_production_output/tfim_h_*_summary.json')):
    with open(filename) as f:
        data = json.load(f)
        h = data['config']['h']
        evals = data['eigenvalues']
        gap = evals[1] - evals[0]
        h_values.append(h)
        gaps.append(gap)

plt.figure(figsize=(8, 6))
plt.semilogy(h_values, gaps, 'o-', linewidth=2, markersize=8)
plt.xlabel('Transverse Field h/J', fontsize=14)
plt.ylabel('Energy Gap Δ', fontsize=14)
plt.title('TFIM Quantum Phase Transition', fontsize=16)
plt.axvline(x=1.0, color='red', linestyle='--', label='Critical point h=J')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tfim_gap.png', dpi=300)
plt.show()
```

---

## Integration with Analysis Pipeline

### Export to Pandas DataFrame

```python
import pandas as pd
import json
import glob

results = []
for filename in glob.glob('ed_production_output/*_summary.json'):
    with open(filename) as f:
        data = json.load(f)
        results.append({
            'filename': filename,
            'model': data['config']['model'],
            'N': data['config']['N'],
            'dim': data['dimensions'],
            'E0': data['eigenvalues'][0],
            'gap': data['eigenvalues'][1] - data['eigenvalues'][0] if len(data['eigenvalues']) > 1 else None,
            'time': data['timing']['total']
        })

df = pd.DataFrame(results)
print(df)
df.to_csv('ed_results_summary.csv', index=False)
```

### Compare with DMRG/QMC

```python
# Your ED result
with open('ed_production_output/my_calc_summary.json') as f:
    ed_data = json.load(f)
    E0_ed = ed_data['eigenvalues'][0]
    N = ed_data['config']['N']

# Your DMRG result (example)
E0_dmrg = -0.4431  # Per site

# Compare
E0_ed_per_site = E0_ed / N
error = abs(E0_ed_per_site - E0_dmrg)

print(f"ED:   E0/N = {E0_ed_per_site:.6f}")
print(f"DMRG: E0/N = {E0_dmrg:.6f}")
print(f"Error: {error:.2e}")
```

---

## Cluster/HPC Usage

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=ed_calc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=ed_%j.out

module load python/3.10
source ~/envs/ed/bin/activate

python exact_diagonalization_production.py \
    --model heisenberg \
    --N 16 \
    --Sz 0 \
    --J 1.0 \
    --output heisenberg_N16_${SLURM_JOB_ID}
```

### PBS Job Script

```bash
#!/bin/bash
#PBS -N ed_calculation
#PBS -l nodes=1:ppn=1
#PBS -l mem=16gb
#PBS -l walltime=02:00:00
#PBS -o ed_${PBS_JOBID}.out
#PBS -e ed_${PBS_JOBID}.err

cd $PBS_O_WORKDIR
source ~/envs/ed/bin/activate

python exact_diagonalization_production.py \
    --model tfim \
    --N 16 \
    --h 1.0 \
    --J 1.0 \
    --boundary periodic \
    --output tfim_N16_${PBS_JOBID}
```

### Array Job for Parameter Scan

```bash
#!/bin/bash
#SBATCH --job-name=ed_scan
#SBATCH --array=1-10
#SBATCH --mem=8GB
#SBATCH --time=01:00:00

# Map array index to parameter
U_values=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 10.0)
U=${U_values[$SLURM_ARRAY_TASK_ID-1]}

python exact_diagonalization_production.py \
    --model hubbard \
    --N 8 \
    --U $U \
    --t 1.0 \
    --N_up 4 \
    --N_dn 4 \
    --output hubbard_U${U}_${SLURM_ARRAY_TASK_ID}
```

---

## Citation Template

If using this code in publications:

### In Methods Section

```
Ground state energies and eigenstates were computed using exact
diagonalization with sparse matrix methods [Ref]. The Hilbert space
was restricted to the appropriate symmetry sectors (Sz conservation
for spins, particle number conservation for fermions). Hamiltonians
were constructed with proper Jordan-Wigner fermionic signs and
validated for Hermiticity (max|H-H†| < 10⁻¹⁰). Eigenvalues were
computed using the Lanczos/ARPACK algorithm as implemented in
SciPy, with convergence tolerance set to 10⁻¹² and eigenvalue
residuals verified to be < 10⁻⁸.
```

### In Code Availability

```
Exact diagonalization calculations were performed using a custom
Python implementation based on NumPy and SciPy sparse linear algebra.
The code includes validation against known exact results and is
available at [URL/DOI]. All calculations are reproducible with
fixed random seeds.
```

### References to Include

```bibtex
@software{ed_production_v2,
  title = {Production-Grade Exact Diagonalization Engine v2.0},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ed_production},
  version = {2.0.0-corrected}
}

@article{scipy,
  title = {SciPy 1.0: fundamental algorithms for scientific computing in Python},
  author = {Virtanen, Pauli and others},
  journal = {Nature Methods},
  volume = {17},
  pages = {261--272},
  year = {2020}
}
```

---

## Getting Help

### Self-Diagnosis

1. **Run validation first**: `--validate`
2. **Check logs**: Look for "✓" or "✗" markers
3. **Verify Hermiticity**: Must appear in logs
4. **Check residuals**: Should be < 1e-8
5. **Test small N**: Always test with N=6-8 first

### Debug Mode

```bash
# Get detailed output
python exact_diagonalization_production.py \
    --model heisenberg \
    --N 8 \
    --Sz 0 \
    2>&1 | tee debug.log

# Check what went wrong
grep -i "error\|warn\|fail" debug.log
```

### Common Error Messages

```
"Hermiticity check FAILED"
→ BUG in code - report immediately

"MemoryError"
→ System too large - reduce N or use sectors

"RuntimeError: Hamiltonian not Hermitian"
→ Construction bug - check model parameters

"ValueError: k >= dim"
→ Too many eigenvalues requested - reduce --k

"Empty basis generated"
→ Impossible quantum numbers - check --Sz, --N_up, --N_dn
```

---

## Next Steps

### For Learning

1. ✅ Run validation suite
2. ✅ Try all three models (heisenberg, tfim, hubbard)
3. ✅ Do a small parameter scan
4. ✅ Plot the results
5. ✅ Compare with known results (Bethe ansatz, etc.)

### For Research

1. ✅ Validate on your specific problem
2. ✅ Test convergence with system size
3. ✅ Compare with other methods (DMRG, QMC)
4. ✅ Check finite-size effects
5. ✅ Document all parameters in supplementary material

### For Production

1. ✅ Write job submission scripts
2. ✅ Set up output organization
3. ✅ Create analysis pipeline
4. ✅ Implement checkpointing for long runs
5. ✅ Benchmark on your cluster

---

## Pro Tips

💡 **Always validate first** - catches 90% of issues  
💡 **Use symmetry sectors** - 5-10× speedup  
💡 **Start small** - test with N=8-10 before N=16  
💡 **Save everything** - disk is cheap, redoing calculations is expensive  
💡 **Check logs** - they tell you if something's wrong  
💡 **Compare methods** - cross-validate with DMRG/QMC  
💡 **Document parameters** - future you will thank you

---

**Last Updated**: October 2025  
**Version**: 2.0.0-corrected
