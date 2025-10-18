# ed_bench/krylov_logger.py
# Usage: import krylov_logger; logger = krylov_logger.Logger('krylov_log.csv'); logger.log(iter, ritz_val, residual)
import csv, time


class Logger:
    def __init__(self, fname="krylov_log.csv"):
        self.fname = fname
        self.start = time.time()
        self.csvfile = open(self.fname, "w", newline="")
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(["iteration", "ritz_val", "residual", "elapsed_s"])
        self.csvfile.flush()

    def log(self, iteration, ritz_val, residual):
        elapsed = time.time() - self.start
        self.writer.writerow([iteration, float(ritz_val), float(residual), elapsed])
        self.csvfile.flush()

    def close(self):
        self.csvfile.close()
