# Quick Reference Card - ED v2.0

## 🚀 Essential Commands (Copy-Paste)

### Validation (ALWAYS RUN FIRST)

```bash
python exact_diagonalization_production.py --validate
```

### Basic Calculations

```bash
# Heisenberg
python exact_diagonalization_production.py --model heisenberg --N 12 --Sz 0

# TFIM
python exact_diagonalization_production.py --model tfim --N 14 --h 1.0 --J 1.0

# Hubbard
python exact_diagonalization_production.py --model hubbard --N 8 --U 4.0 --N_up 4 --N_dn 4
```

### Benchmarking

```bash
# Quick benchmark
python benchmark_ed_scaling_v2.py --models tfim heisenberg --sizes 8 10 12

# With safety limits
python benchmark_ed_scaling_v2.py --sizes 8 10 12 --max-dim 20000 --max-memory-gb 8
```

---

## 📊 Parameter Quick Reference

### Models

- `--model heisenberg` → XXZ Heisenberg
- `--model tfim` → Transverse-field Ising
- `--model hubbard` → Fermi-Hubbard

### Key Parameters

- `--N <int>` → System size
- `--Sz <int>` → Sz sector (Heisenberg, units of 1/2)
- `--N_up <int>` → Spin-up fermions (Hubbard)
- `--N_dn <int>` → Spin-down fermions (Hubbard)
- `--J <float>` → Exchange coupling
- `--h <float>` → Transverse field
- `--U <float>` → Hubbard U
- `--t <float>` → Hopping

### Control

- `--method eigsh|lanczos|full` → Solver
- `--k <int>` → Number of eigenvalues (default: 6)
- `--boundary open|periodic` → Boundary conditions
- `--output <name>` → Output prefix

---

## 🔍 Quick Checks

### Is It Working?

```bash
# Check output exists
ls ed_production_output/*.json

# Check ground state energy
jq '.eigenvalues[0]' ed_production_output/my_run_summary.json

# Check Hermiticity passed
grep "Hermiticity validated" <output>.log
```

### Quality Checks

```bash
# Hermiticity error
jq '.validation.max_asymmetry' ed_production_output/*.json

# Eigenvalue residuals
jq '.validation.max_residual' ed_production_output/*.json

# Both should be < 1e-8 for publication quality
```

---

## 📈 System Size Limits

| N   | Heisenberg (Sz=0) | TFIM (full) | Hubbard (half) | Device      |
| --- | ----------------- | ----------- | -------------- | ----------- |
| 8   | dim=70            | dim=256     | dim~70         | Laptop      |
| 10  | dim=252           | dim=1024    | dim~2000       | Laptop      |
| 12  | dim=924           | dim=4096    | dim~20000      | Workstation |
| 14  | dim=3432          | dim=16384   | —              | Workstation |
| 16  | dim=12870         | dim=65536   | —              | Server      |
| 18  | dim=48620         | —           | —              | HPC         |

**Rule**: Stop ED at N~18-20, use DMRG beyond

---

## 🚨 Error Messages

| Message                  | Meaning                 | Fix                             |
| ------------------------ | ----------------------- | ------------------------------- |
| Hermiticity check FAILED | Bug in code             | Report immediately              |
| MemoryError              | Too large               | Reduce N or use --max-memory-gb |
| Using v1.0 ED module     | Wrong version           | Use v2.0-corrected              |
| Large residual           | Poor convergence        | Increase --tol                  |
| Empty basis              | Invalid quantum numbers | Check --Sz, --N_up, --N_dn      |

---

## 📁 Output Files

```
ed_production_output/
├── <prefix>.h5              # HDF5: full data
├── <prefix>_summary.json    # Human-readable
└── <prefix>_observables.npz # Observable arrays

bench_outputs/
├── benchmark_results.csv    # Spreadsheet data
├── benchmark_summary.txt    # Report
└── bench_*.json            # Per-run details
```

---

## 💡 Pro Tips

✅ **Always validate first** (`--validate`)  
✅ **Test at N=8** before scaling up  
✅ **Use symmetry sectors** (5-10× speedup)  
✅ **Check logs** for ✓ symbols  
✅ **GPU for N=12-14** (--use-gpu)

❌ **Never skip validation**  
❌ **Never use v1.0 for Hubbard**  
❌ **Never ignore Hermiticity failures**  
❌ **Never trust large residuals** (>1e-6)

---

## 🔗 Documentation Map

- **Getting started** → QUICKSTART.md
- **Full reference** → README.md
- **What was fixed** → CORRECTIONS_v2.0.md
- **Benchmarking** → BENCHMARK_README.md
- **This card** → Keep handy!

---

## ⚡ One-Liners

```bash
# Validate
python exact_diagonalization_production.py --validate && echo "✓ READY"

# Quick test
python exact_diagonalization_production.py --model heisenberg --N 8 --Sz 0 --output test

# Check if Hermitian
jq '.validation.is_hermitian' ed_production_output/test_summary.json

# Extract E0
jq '.eigenvalues[0]' ed_production_output/test_summary.json

# Benchmark with limits
python benchmark_ed_scaling_v2.py --sizes 8 10 12 --max-memory-gb 8 --max-dim 10000
```

---

**Version**: 2.0.0-corrected  
**Print this card** and keep it next to your terminal! 🖨️
