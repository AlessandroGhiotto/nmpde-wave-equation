#!/usr/bin/env python3
"""
Convergence sweep for the wave-equation solver.

Runs all (scheme, Nel, R, dt) combinations with CFL-safe filtering for
explicit methods, and collects the convergence CSV produced by the C++ binary.

Usage (local):
    python3 convergence_sweep.py --nprocs 4

Usage (cluster):
    refere to convergence_all.pbs
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ============================================================
# CLI arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="Convergence sweep for wave-equation solver"
)
parser.add_argument(
    "--nprocs", type=int, default=4, help="Number of MPI processes (default: 4)"
)
parser.add_argument(
    "--launcher",
    default=os.environ.get("MPI_LAUNCHER", "mpirun"),
    help="MPI launcher command (default: $MPI_LAUNCHER or mpirun)",
)
parser.add_argument(
    "--mpi-arg", action="append", default=[], help="Extra MPI launcher arg (repeatable)"
)
parser.add_argument("--bind-to-core", action="store_true", default=True)
parser.add_argument("--no-bind-to-core", action="store_false", dest="bind_to_core")
parser.add_argument(
    "--use-pbs-nodefile",
    action="store_true",
    help="Pass $PBS_NODEFILE as hostfile to the launcher",
)
parser.add_argument(
    "--job-id",
    default=os.environ.get("PBS_JOBID", ""),
    help="Job identifier for output filenames (default: $PBS_JOBID or empty)",
)

# Discretization grid (overridable from CLI)
parser.add_argument("--nel", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
parser.add_argument("--r", type=int, nargs="+", default=[1, 2], dest="R_values")
parser.add_argument(
    "--dt",
    type=float,
    nargs="+",
    default=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
)
parser.add_argument("--T", type=float, default=1.0)
parser.add_argument(
    "--schemes",
    nargs="+",
    default=["theta-0.0", "theta-0.5", "theta-1.0", "newmark-0.00", "newmark-0.25"],
    help="Which schemes to run (default: all five)",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=600,
    help="Per-run timeout in seconds (default: 600). Kills runs that hang/diverge.",
)
parser.add_argument(
    "--cfl-safety",
    type=float,
    default=0.9,
    help="CFL safety factor for explicit methods (default: 0.9)",
)

args = parser.parse_args()

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_PARAM_PATH = (PROJECT_ROOT / "parameters" / "standing-mode-wsol.json").resolve()
BINARY_THETA = (PROJECT_ROOT / "build" / "main-theta").resolve()
BINARY_NEWMARK = (PROJECT_ROOT / "build" / "main-newmark").resolve()

# The C++ binary writes convergence data to:
#   ../results/{problem_name}/convergence.csv   (relative to CWD)
# problem_name = "{theta|newmark}-{param_file_stem}"
# We use a fixed param file name "conv-params.json" so that:
#   theta   -> ../results/theta-conv-params/convergence.csv
#   newmark -> ../results/newmark-conv-params/convergence.csv
PARAM_STEM = "conv-params"

# ============================================================
# Scheme definitions
# ============================================================
SCHEME_DEFS = {
    "theta-0.0": {
        "binary": BINARY_THETA,
        "overrides": {"Theta": "0.0"},
        "explicit": True,
    },
    "theta-0.5": {
        "binary": BINARY_THETA,
        "overrides": {"Theta": "0.5"},
        "explicit": False,
    },
    "theta-1.0": {
        "binary": BINARY_THETA,
        "overrides": {"Theta": "1.0"},
        "explicit": False,
    },
    "newmark-0.00": {
        "binary": BINARY_NEWMARK,
        "overrides": {"Beta": "0.0", "Gamma": "0.5"},
        "explicit": True,
    },
    "newmark-0.25": {
        "binary": BINARY_NEWMARK,
        "overrides": {"Beta": "0.25", "Gamma": "0.5"},
        "explicit": False,
    },
}


# ============================================================
# CFL logic
# ============================================================
def cfl_limit(nel: int, r: int, c: float = 1.0) -> float:
    """
    Conservative CFL limit for explicit time integration on a 2D simplex mesh.

    For FE_SimplexP<r> on a structured triangular mesh with edge length h = 1/Nel
    """
    h = 1.0 / nel
    p_factor = 1.0 if r == 1 else 4.0
    return args.cfl_safety * h / (c * math.sqrt(2) * p_factor)


def is_cfl_safe(scheme_name: str, nel: int, r: int, dt: float) -> bool:
    """Check whether a dt value is CFL-safe for the given scheme."""
    if not SCHEME_DEFS[scheme_name]["explicit"]:
        return True  # implicit methods have no CFL restriction
    return dt <= cfl_limit(nel, r)


# ============================================================
# Parameter file & runner
# ============================================================
def load_base_params() -> dict:
    with BASE_PARAM_PATH.open() as f:
        return json.load(f)


def write_param_file(
    base: dict, nel: int, r: int, dt: float, T: float, overrides: dict, out_path: Path
):
    """Write a parameter JSON file for one run."""
    params = dict(base)
    params["Nel"] = str(nel)
    params["R"] = str(r)
    params["Dt"] = str(dt)
    params["T"] = str(T)
    # Disable heavy I/O for convergence sweeps
    params["Save Solution"] = False
    params["Enable Logging"] = False
    params["Log Every"] = 0
    params.update(overrides)
    out_path.write_text(json.dumps(params, indent=2))


def build_mpi_cmd(binary: Path, param_file: Path, nel: int = None) -> list[str]:
    """Build the MPI launch command.

    For very small meshes (Nel <= 10), uses 1 process to avoid MPI overhead/errors.
    """
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    # Use single process for tiny meshes
    nprocs = 1 if (nel is not None and nel <= 10) else args.nprocs
    cmd = [args.launcher, "-np", str(nprocs)]

    # PBS nodefile integration
    if args.use_pbs_nodefile:
        pbs_nodefile = os.environ.get("PBS_NODEFILE")
        if pbs_nodefile and Path(pbs_nodefile).exists():
            hosts = list(dict.fromkeys(Path(pbs_nodefile).read_text().splitlines()))
            host = hosts[0] if hosts else None
            if host:
                clean_hf = Path(f"/tmp/hostfile_conv_{os.getpid()}")
                clean_hf.write_text(f"{host} slots={args.nprocs}\n")
                cmd += ["--hostfile", str(clean_hf)]

    if args.bind_to_core:
        cmd += ["--bind-to", "core", "--map-by", "socket"]

    cmd += list(args.mpi_arg)
    cmd += [str(binary), str(param_file)]
    return cmd


def run_single(
    binary: Path, param_file: Path, tag: str, logs_dir: Path, nel: int = None
):
    """Run one simulation. Returns (returncode, elapsed_seconds)."""
    cmd = build_mpi_cmd(binary, param_file, nel)
    stdout_path = logs_dir / f"{tag}.out"
    stderr_path = logs_dir / f"{tag}.err"

    print(f"  [RUN] {' '.join(cmd[-3:])}")
    t0 = time.perf_counter()
    try:
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            r = subprocess.run(cmd, stdout=out, stderr=err, timeout=args.timeout)
        elapsed = time.perf_counter() - t0
        return r.returncode, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  [TIMEOUT] killed after {elapsed:.1f}s")
        return -1, elapsed


# ============================================================
# Main sweep
# ============================================================
def main():
    base = load_base_params()

    # Validate schemes
    for s in args.schemes:
        if s not in SCHEME_DEFS:
            print(f"Unknown scheme: {s}. Available: {list(SCHEME_DEFS.keys())}")
            sys.exit(1)

    # Clean old convergence CSVs so we start fresh
    results_base = Path("../results")
    for prefix in ("theta", "newmark"):
        csv_path = results_base / f"{prefix}-{PARAM_STEM}" / "convergence.csv"
        if csv_path.exists():
            csv_path.unlink()
            print(f"Removed old {csv_path}")

    # Prepare directories
    logs_dir = Path.cwd() / "convergence-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build the run plan
    plan = []
    for scheme_name in args.schemes:
        sdef = SCHEME_DEFS[scheme_name]
        for nel in sorted(args.nel):
            for r in sorted(args.R_values):
                cfl = cfl_limit(nel, r)
                for dt in sorted(args.dt, reverse=True):
                    if is_cfl_safe(scheme_name, nel, r, dt):
                        plan.append((scheme_name, nel, r, dt))
                    else:
                        pass

    total = len(plan)
    print(f"\n{'='*60}")
    print(f"Convergence sweep: {total} runs")
    print(f"  Schemes: {args.schemes}")
    print(f"  Nel:     {args.nel}")
    print(f"  R:       {args.R_values}")
    print(f"  dt:      {args.dt}")
    print(f"  T:       {args.T}")
    print(f"  nprocs:  {args.nprocs}")
    print(f"  timeout: {args.timeout}s per run")
    print(f"{'='*60}\n")

    # Run log CSV
    job_suffix = f"-{args.job_id}" if args.job_id else ""
    runlog_path = Path(f"convergence-runlog{job_suffix}.csv")
    with runlog_path.open("w") as logf:
        logf.write("scheme,Nel,R,dt,T,returncode,elapsed_s,cfl_limit\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            param_file = Path(tmpdir) / f"{PARAM_STEM}.json"

            for i, (scheme_name, nel, r, dt) in enumerate(plan, 1):
                sdef = SCHEME_DEFS[scheme_name]
                cfl = cfl_limit(nel, r) if sdef["explicit"] else float("inf")

                tag = f"{scheme_name}_Nel{nel}_R{r}_dt{dt}"
                print(
                    f"[{i}/{total}] {tag}"
                    + (f"  (CFL={cfl:.6f})" if sdef["explicit"] else "")
                )

                # Write params
                write_param_file(
                    base, nel, r, dt, args.T, sdef["overrides"], param_file
                )

                # Run
                code, elapsed = run_single(
                    sdef["binary"], param_file, tag, logs_dir, nel
                )
                status = (
                    "OK"
                    if code == 0
                    else ("TIMEOUT" if code == -1 else f"FAIL({code})")
                )
                print(f"  -> {status} in {elapsed:.1f}s")

                logf.write(
                    f"{scheme_name},{nel},{r},{dt},{args.T},{code},{elapsed:.3f},{cfl:.8f}\n"
                )
                logf.flush()

    # Merge convergence CSVs produced by the binary
    merged_path = Path(f"convergence-results{job_suffix}.csv")
    header_written = False
    with merged_path.open("w") as out:
        for prefix in ("theta", "newmark"):
            csv_path = results_base / f"{prefix}-{PARAM_STEM}" / "convergence.csv"
            if csv_path.exists():
                with csv_path.open() as inp:
                    for line_no, line in enumerate(inp):
                        if line_no == 0:
                            if not header_written:
                                out.write(line)
                                header_written = True
                        else:
                            out.write(line)

    print(f"\n{'='*60}")
    print(f"Done. Merged convergence results: {merged_path}")
    print(f"Run log: {runlog_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
