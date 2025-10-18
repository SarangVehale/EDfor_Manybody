#!/usr/bin/env bash
# ed_bench/run_benchmarks.sh
# Usage: ./run_benchmarks.sh <model> <sector> <output-dir>
set -euo pipefail

MODEL=${1:-heisenberg} # model name: heisenberg/hubbard/ising
SECTOR=${2:-Sz=0}      # sector label used by your ED driver
OUTDIR=${3:-bench_results}
mkdir -p "${OUTDIR}"
TIMINGS_CSV="${OUTDIR}/timings_${MODEL}_${SECTOR//=/}.csv"

# CSV header
echo "N,hilbert_dim,wall_time_s,user_time_s,sys_time_s,max_rss_kb,exit_status,logfile" >"${TIMINGS_CSV}"

# sizes to attempt - edit as needed
SIZES=(10 12 14 16 18 20 22 24)

for N in "${SIZES[@]}"; do
	echo "=== Running N=${N} ==="
	LOG="${OUTDIR}/${MODEL}_N${N}_log.txt"
	TIMELOG="${OUTDIR}/${MODEL}_N${N}_time.txt"

	# Compute hilbert dimension if your driver can't report it:
	# For spin-1/2 full space: 2^N; if using Sz sector, compute binomial coefficient
	# We'll attempt to let the ED program print dims; otherwise precompute:
	HILB="$(
		python3 - <<PY
import math,sys
N=int(${N})
# default: S^z = 0 sector (even N only)
from math import comb
if ${N}%2==0:
    print(comb(N,N//2))
else:
    print(2**N)
PY
	)"

	# run the ed solver under /usr/bin/time
	# Replace the command below with the actual ED call. The call must write meaningful stdout or optionally print hilbert dim
	/usr/bin/time -v python3 run_ed.py --model "${MODEL}" --L "${N}" --sector "${SECTOR}" >"${LOG}" 2>"${TIMELOG}" || echo "run failed for N=${N}"

	# parse /usr/bin/time -v fields
	WALL=$(grep "Elapsed (wall clock) time" "${TIMELOG}" | awk -F: '{gsub(/,/,"",$2); print $2}' | awk '{$1=$1; print}')
	# convert HH:MM:SS to seconds if necessary
	if [[ "${WALL}" == *:* ]]; then
		IFS=':' read -r h m s <<<"${WALL}"
		WALL_SEC=$(
			python3 - <<PY
h=int(${h}); m=int(${m}); s=float(${s})
print(h*3600 + m*60 + s)
PY
		)
	else
		WALL_SEC=${WALL}
	fi

	USER_TIME=$(grep "User time (seconds)" "${TIMELOG}" | awk -F: '{print $2}' | tr -d ' ')
	SYS_TIME=$(grep "System time (seconds)" "${TIMELOG}" | awk -F: '{print $2}' | tr -d ' ')
	MAX_RSS=$(grep "Maximum resident set size" "${TIMELOG}" | awk -F: '{gsub(/ /,"",$2); print $2}')

	# exit status: try to detect "Command exited" or missing
	EXIT_STATUS=0
	if grep -q "Command exited with non-zero status" "${TIMELOG}"; then
		EXIT_STATUS=1
	fi

	echo "${N},${HILB},${WALL_SEC},${USER_TIME},${SYS_TIME},${MAX_RSS},${EXIT_STATUS},${LOG}" >>"${TIMINGS_CSV}"
done

echo "Benchmarks complete. CSV saved to ${TIMINGS_CSV}"
