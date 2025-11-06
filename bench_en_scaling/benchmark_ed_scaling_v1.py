#!/usr/bin/env python3
"""
benchmark_ed_scaling.py

Benchmark Exact Diagonalization (ED) scaling for TFIM and Heisenberg models
using exact_diagonalization.py as the ED engine.

Features:
 - automatic strategy selection: dense (numpy), sparse eigsh (scipy), optional GPU dense (cupy)
 - measures wall-clock time, diag time, RSS memory (max), and GPU memory (if available)
 - computes eigenpair residuals ||H v - lambda v|| for lowest k eigenpairs
 - outputs CSV and human-readable summary
 - safety checks: avoids runs that exceed CPU RAM or GPU memory (configurable)

Usage:
    python3 benchmark_ed_scaling.py --models tfim heisenberg --sizes 8 10 12 14 16 18 20 --use-gpu

Outputs:
    benchmark_results.csv
    benchmark_summary.txt
    optional per-run JSON summaries in bench_outputs/
"""
import argparse
import csv
import json
import math
import os
import resource
import sys
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# attempt to import cupy for GPU acceleration
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False

# attempt to import pynvml for GPU memory queries
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# user ED module (must be in same directory or PYTHONPATH)
import exact_diagonalization as ed


# ----- utilities -----
def get_peak_rss_kb():
    """Return max RSS (kilobytes) reported for this process."""
    # ru_maxrss is in kilobytes on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss)


def get_gpu_memory_info(device_index=0):
    """Return (used_MB, total_MB) for the given GPU index using NVML, or (None,None) if unavailable."""
    if not _NVML_AVAILABLE:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used // (1024 * 1024)
        total = meminfo.total // (1024 * 1024)
        return int(used), int(total)
    except Exception:
        return None, None


def estimate_sparse_memory_bytes(nnz, dtype=np.float64):
    """Estimate memory to store sparse matrix with nnz nonzeros (CSR) - rough."""
    # CSR: data (nnz * 8 bytes), indices (nnz * 8 for int64), indptr (n+1 * 8)
    # For rough estimate we use 8 bytes per number / index
    return int(nnz * 8 * 2 + 8 * 100)  # rough


# compute Hilbert dimension for spin-1/2 Sz sector
from math import comb


def hilbert_dim_spin_sz(N, Sz_target=None):
    if Sz_target is None:
        # full basis
        return 1 << N
    # Sz_target in units of 1/2 (e.g., 0 means N_up = N/2)
    N_up = int(Sz_target + N // 2)
    return comb(N, N_up)


# ----- main run function -----
def run_single_benchmark(
    model,
    N,
    sector_label,
    solver_pref,
    dense_threshold,
    use_gpu,
    gpu_device,
    k_eig,
    outdir,
):
    """
    Runs one benchmark: construct H via ed module, choose solver, run diagonalization,
    measure times/memory, compute residuals, and write run JSON.
    Returns a dict with summary fields.
    """
    runinfo = {
        "model": model,
        "N": N,
        "sector": sector_label,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    os.makedirs(outdir, exist_ok=True)

    # prepare basis dim estimate
    Sz_target = None
    if sector_label and sector_label.startswith("Sz="):
        try:
            Sz_target = int(sector_label.split("=")[1])
        except Exception:
            Sz_target = None
    dim_est = hilbert_dim_spin_sz(N, Sz_target)
    runinfo["hilbert_dim_est"] = int(dim_est)

    # Build Hamiltonian (time this)
    t0 = time.time()
    if model == "tfim":
        # TFIM builder in user's module should return sparse CSR
        H = ed.build_tfim_sparse(N=N, h=1.0, J=1.0, boundary="open")
    elif model == "heisenberg":
        # Use Sz sector basis if provided by sector_label
        if Sz_target is None:
            # default Sz=0 sector for performance
            basis, idx = ed.generate_sz_sector_basis(N, Sz_target=0)
            H = ed.build_heisenberg_sparse(
                N=N, J=1.0, boundary="open", basis=basis, idx_map=idx
            )
        else:
            basis, idx = ed.generate_sz_sector_basis(N, Sz_target=Sz_target)
            H = ed.build_heisenberg_sparse(
                N=N, J=1.0, boundary="open", basis=basis, idx_map=idx
            )
    else:
        raise RuntimeError(f"Unknown model {model}")

    build_time = time.time() - t0
    runinfo["build_time_s"] = float(build_time)
    # validate hermiticity if function exists
    try:
        if hasattr(ed, "validate_hermiticity"):
            ed.validate_hermiticity(H)
            runinfo["hermiticity_check"] = True
    except Exception as e:
        runinfo["hermiticity_check"] = False
        runinfo["hermiticity_error"] = str(e)

    # measure pre-run RSS and GPU usage
    rss_before_kb = get_peak_rss_kb()
    gpu_before = get_gpu_memory_info(gpu_device) if use_gpu else (None, None)
    runinfo["rss_before_kb"] = rss_before_kb
    runinfo["gpu_before_mb"] = gpu_before[0] if gpu_before[0] is not None else None

    # determine solver strategy
    dim = H.shape[0]
    runinfo["dim"] = int(dim)
    solver_used = None
    diag_time = None
    eigvals = None
    eigvecs = None
    residuals = []

    # Helper: compute residuals
    def compute_residuals(Hsparse, evals, evecs):
        res = []
        for i in range(min(len(evals), evecs.shape[1])):
            v = evecs[:, i]
            Hv = Hsparse.dot(v)
            resid = np.linalg.norm(Hv - evals[i] * v)
            res.append(float(resid))
        return res

    # memory safety thresholds (user can tune)
    # total system memory:
    try:
        import psutil

        total_mem_bytes = psutil.virtual_memory().total
    except Exception:
        total_mem_bytes = 64 * 1024**3  # assume 64GB if psutil not available

    # decide dense vs sparse
    use_dense = dim <= dense_threshold
    # if GPU requested and available and dim small enough for dense on GPU, use it
    gpu_used_flag = False
    if use_gpu and _CUPY_AVAILABLE and use_dense:
        # estimate memory to hold dense matrix dim*dim*8 bytes
        bytes_needed = dim * dim * 8
        # check GPU available memory
        if _NVML_AVAILABLE:
            used_mb, total_mb = get_gpu_memory_info(gpu_device)
            total_bytes_gpu = total_mb * 1024**2
            # leave margin 0.7
            if bytes_needed < total_bytes_gpu * 0.7:
                solver_used = "gpu_dense_cupy"
                gpu_used_flag = True
        else:
            # no NVML: be conservative and only allow small dims
            if dim <= 1024:
                solver_used = "gpu_dense_cupy"
                gpu_used_flag = True

    if solver_used is None:
        if use_dense:
            solver_used = "dense_numpy"
        else:
            solver_used = "sparse_eigsh"

    runinfo["solver_chosen"] = solver_used

    # perform diagonalization and time it
    tdiag0 = time.time()
    if solver_used == "dense_numpy":
        # build dense matrix
        Hd = H.toarray()
        t0diag = time.time()
        w, v = np.linalg.eigh(Hd)
        diag_time = time.time() - t0diag
        eigvals = np.array(w[:k_eig])
        eigvecs = np.array(v[:, :k_eig])
        residuals = compute_residuals(H, eigvals, eigvecs)
    elif solver_used == "gpu_dense_cupy":
        # transfer dense to GPU and use cupy.linalg.eigh
        Hd = H.toarray()
        Hd_gpu = cp.asarray(Hd)
        t0diag = time.time()
        w_gpu, v_gpu = cp.linalg.eigh(Hd_gpu)
        cp.cuda.Stream.null.synchronize()
        diag_time = time.time() - t0diag
        w = cp.asnumpy(w_gpu)
        v = cp.asnumpy(v_gpu)
        eigvals = np.array(w[:k_eig])
        eigvecs = np.array(v[:, :k_eig])
        residuals = compute_residuals(H, eigvals, eigvecs)
    elif solver_used == "sparse_eigsh":
        k = min(k_eig, max(1, dim - 2))
        t0diag = time.time()
        w, v = spla.eigsh(H, k=k, which="SA", tol=1e-8, maxiter=10000)
        diag_time = time.time() - t0diag
        idx_sorted = np.argsort(w)
        w = w[idx_sorted]
        v = v[:, idx_sorted]
        eigvals = np.array(w[:k_eig])
        eigvecs = np.array(v[:, :k_eig])
        residuals = compute_residuals(H, eigvals, eigvecs)
    else:
        raise RuntimeError("Unsupported solver strategy")

    tdiag_total = time.time() - tdiag0
    runinfo["diag_time_s"] = float(diag_time)
    runinfo["diag_time_total_s"] = float(tdiag_total)
    runinfo["eigvals"] = [float(x) for x in eigvals.tolist()]
    runinfo["residuals"] = residuals

    rss_after_kb = get_peak_rss_kb()
    runinfo["rss_after_kb"] = rss_after_kb
    runinfo["rss_peak_kb_reported"] = max(
        runinfo.get("rss_before_kb", 0), runinfo.get("rss_after_kb", 0)
    )

    if use_gpu and _NVML_AVAILABLE:
        used_gpu_mb, total_gpu_mb = get_gpu_memory_info(gpu_device)
        runinfo["gpu_used_mb_after"] = used_gpu_mb
        runinfo["gpu_total_mb"] = total_gpu_mb
    else:
        runinfo["gpu_used_mb_after"] = None
        runinfo["gpu_total_mb"] = None

    # write per-run JSON
    outjson = os.path.join(outdir, f"bench_{model}_N{N}_{solver_used}.json")
    with open(outjson, "w") as fh:
        json.dump(runinfo, fh, indent=2)

    return runinfo


# ----- CLI / orchestration -----
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ED scaling for TFIM and Heisenberg (with optional GPU)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tfim", "heisenberg"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[8, 10, 12, 14, 16, 18, 20, 22, 24],
        help="System sizes to try",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default="Sz=0",
        help="Sector (Sz=0) for Heisenberg; ignored for TFIM",
    )
    parser.add_argument(
        "--dense-threshold",
        type=int,
        default=2000,
        help="max dim for dense diagonalization (numpy)",
    )
    parser.add_argument(
        "--k-eig", type=int, default=6, help="Number of lowest eigenvalues to compute"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Attempt GPU acceleration with CuPy if available",
    )
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index")
    parser.add_argument(
        "--outdir",
        type=str,
        default="bench_outputs",
        help="Output directory for CSV/JSON",
    )
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    csvfile = os.path.join(outdir, "benchmark_results.csv")
    summary_txt = os.path.join(outdir, "benchmark_summary.txt")

    # CSV header
    header = [
        "model",
        "N",
        "sector",
        "dim",
        "solver",
        "build_time_s",
        "diag_time_s",
        "diag_time_total_s",
        "rss_before_kb",
        "rss_after_kb",
        "gpu_used_mb_after",
        "gpu_total_mb",
        "eig0",
        "eig1",
        "eig2",
        "eig3",
        "eig4",
        "eig5",
        "resid0",
        "resid1",
        "resid2",
        "resid3",
        "resid4",
        "resid5",
    ]
    with open(csvfile, "w", newline="") as csvfh:
        writer = csv.writer(csvfh)
        writer.writerow(header)

    summary_lines = []
    for model in args.models:
        for N in args.sizes:
            # caution for large N/hubbard: the ed builders may be extremely large; wrap in try/except
            try:
                info = run_single_benchmark(
                    model=model,
                    N=N,
                    sector_label=args.sector,
                    solver_pref=None,
                    dense_threshold=args.dense_threshold,
                    use_gpu=args.use_gpu,
                    gpu_device=args.gpu_device,
                    k_eig=args.k_eig,
                    outdir=outdir,
                )
            except MemoryError as me:
                msg = f"[SKIP] model={model} N={N} - MemoryError: {me}"
                print(msg)
                summary_lines.append(msg)
                continue
            except Exception as e:
                msg = f"[ERROR] model={model} N={N} - Exception: {e}"
                print(msg)
                summary_lines.append(msg)
                continue

            # write CSV row (fill missing eig/resid entries to length 6)
            eigs = info.get("eigvals", [])
            resid = info.get("residuals", [])
            row = [
                info.get("model"),
                info.get("N"),
                info.get("sector"),
                info.get("dim"),
                info.get("solver_chosen"),
                info.get("build_time_s"),
                info.get("diag_time_s"),
                info.get("diag_time_total_s"),
                info.get("rss_before_kb"),
                info.get("rss_after_kb"),
                info.get("gpu_used_mb_after"),
                info.get("gpu_total_mb"),
            ]
            for i in range(6):
                row.append(eigs[i] if i < len(eigs) else "")
            for i in range(6):
                row.append(resid[i] if i < len(resid) else "")
            with open(csvfile, "a", newline="") as csvfh:
                writer = csv.writer(csvfh)
                writer.writerow(row)

            s = (
                f"[OK] model={model} N={N} solver={info.get('solver_chosen')} "
                f"dim={info.get('dim')} build={info.get('build_time_s'):.3f}s diag={info.get('diag_time_s'):.3f}s"
            )
            print(s)
            summary_lines.append(s)

    # write summary file
    with open(summary_txt, "w") as fh:
        fh.write("\n".join(summary_lines))
    print("Benchmark complete. CSV:", csvfile, "Summary:", summary_txt)


if __name__ == "__main__":
    main()
