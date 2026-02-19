#!/usr/bin/env python3
"""
Dissipation & dispersion sweep for the wave-equation solver.

Uses the standing-mode test case  u(x,y,t) = cos(sqrt(2)*pi*t)*sin(pi*x)*sin(pi*y)
on [0,1]^2 with zero forcing/BCs.  The exact solution has:
    omega_exact = sqrt(2)*pi   (frequency)
    alpha_exact = 0            (zero damping)
    E(t) = const               (energy conservation)

We fix a fine spatial mesh (to isolate temporal error) and sweep over dt
for each scheme.  The C++ binary logs energy.csv (every step) so we can
measure:
    - Dissipation  via  E(T)/E(0)
    - Dispersion   via  relative L2 error growth pattern

Usage (local):
    python3 dissipation_dispersion_sweep.py --nprocs 4

Usage (cluster):
    python3 dissipation_dispersion_sweep.py --nprocs 4 --use-pbs-nodefile
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
    description="Dissipation/dispersion sweep for wave-equation solver"
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

# Discretization grid
parser.add_argument(
    "--nel",
    type=int,
    default=60,
    help="Elements per side for implicit schemes (fine enough to isolate temporal error)",
)
parser.add_argument(
    "--nel-explicit",
    type=int,
    default=60,
    help="Elements per side for explicit (conditionally stable) schemes "
    "(default: 60; smaller mesh relaxes the CFL constraint, "
    "ensuring stable results for theta=0 and Newmark CD)",
)
parser.add_argument("--r", type=int, default=1, help="FE polynomial degree")
parser.add_argument(
    "--dt",
    type=float,
    nargs="+",
    default=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005],
)
parser.add_argument(
    "--T", type=float, default=5.0, help="Final time (long enough for several periods)"
)
parser.add_argument(
    "--schemes",
    nargs="+",
    default=["theta-0.0", "theta-0.5", "theta-1.0", "newmark-0.00", "newmark-0.25"],
    help="Which schemes to run (default: all five)",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=3000,
    help="Per-run timeout in seconds (default: 3000)",
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

# Param file stem used for output directory naming
PARAM_STEM = "dissdisp-params"

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
# CFL logic  (same as convergence_sweep.py)
# ============================================================
def cfl_limit(nel: int, r: int, c: float = 1.0) -> float:
    h = 1.0 / nel
    p_factor = 1.0 if r == 1 else 4.0
    return args.cfl_safety * h / (c * math.sqrt(2) * p_factor)


def is_cfl_safe(scheme_name: str, nel: int, r: int, dt: float) -> bool:
    if not SCHEME_DEFS[scheme_name]["explicit"]:
        return True
    return dt <= cfl_limit(nel, r)


# ============================================================
# Parameter file helpers
# ============================================================
def load_base_params() -> dict:
    with BASE_PARAM_PATH.open() as f:
        return json.load(f)


def write_param_file(
    base: dict, nel: int, r: int, dt: float, T: float, overrides: dict, out_path: Path
):
    """Write a parameter JSON for one dissipation/dispersion run.

    Key differences from convergence sweep:
      - Save Solution = False  (no VTU output, save disk space)
      - Enable Logging = True  (we need energy + error CSV)
      - Log Every = 1          (log every single timestep for max resolution)
    """
    params = dict(base)
    params["Nel"] = str(nel)
    params["R"] = str(r)
    params["Dt"] = str(dt)
    params["T"] = str(T)
    params["Save Solution"] = False
    params["Enable Logging"] = True
    params["Log Every"] = 1
    params["Print Every"] = max(1, int(1.0 / dt))  # print ~once per second of sim time
    params.update(overrides)
    out_path.write_text(json.dumps(params, indent=2))


def build_mpi_cmd(binary: Path, param_file: Path, nel: int = None) -> list[str]:
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    nprocs = 1 if (nel is not None and nel <= 10) else args.nprocs
    cmd = [args.launcher, "-np", str(nprocs)]

    if args.use_pbs_nodefile:
        pbs_nodefile = os.environ.get("PBS_NODEFILE")
        if pbs_nodefile and Path(pbs_nodefile).exists():
            hosts = list(dict.fromkeys(Path(pbs_nodefile).read_text().splitlines()))
            host = hosts[0] if hosts else None
            if host:
                clean_hf = Path(f"/tmp/hostfile_dissdisp_{os.getpid()}")
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
# Post-processing: extract dissipation/dispersion from energy/error CSVs
# ============================================================
def extract_metrics(results_base: Path, problem_name: str, run_folder: str) -> dict:
    """Extract dissipation/dispersion metrics from a completed run.

    Returns dict with:
      - energy_ratio:       E(T)/E(0)  (1.0 = no dissipation)
      - energy_decay_rate:  (E(0)-E(T))/(E(0)*T)  (0 = no dissipation)
      - max_rel_L2_error:   max relative L2 error over time
      - final_rel_L2_error: relative L2 error at final time
      - final_rel_H1_error: relative H1 error at final time
    """
    import csv

    run_dir = results_base / problem_name / run_folder
    metrics = {}

    # --- Energy CSV ---
    energy_path = run_dir / "energy.csv"
    if energy_path.exists():
        with energy_path.open() as f:
            reader = csv.DictReader(f)
            energies = []
            for row in reader:
                energies.append((float(row["time"]), float(row["energy"])))

        if len(energies) >= 2:
            E0 = energies[0][1]
            ET = energies[-1][1]
            T_actual = energies[-1][0]
            metrics["E0"] = E0
            metrics["ET"] = ET
            metrics["energy_ratio"] = ET / E0 if E0 > 0 else float("nan")
            metrics["energy_decay_rate"] = (
                (E0 - ET) / (E0 * T_actual)
                if (E0 > 0 and T_actual > 0)
                else float("nan")
            )
            # Collect full energy time-series for later analysis
            metrics["energy_times"] = [e[0] for e in energies]
            metrics["energy_values"] = [e[1] for e in energies]

    # --- Error CSV ---
    error_path = run_dir / "error.csv"
    if error_path.exists():
        with error_path.open() as f:
            reader = csv.DictReader(f)
            errors = []
            for row in reader:
                errors.append(
                    {
                        "time": float(row["time"]),
                        "rel_L2": float(row["rel_L2_error"]),
                        "rel_H1": float(row["rel_H1_error"]),
                    }
                )

        if errors:
            metrics["max_rel_L2_error"] = max(e["rel_L2"] for e in errors)
            metrics["final_rel_L2_error"] = errors[-1]["rel_L2"]
            metrics["final_rel_H1_error"] = errors[-1]["rel_H1"]
            # Collect full error time-series
            metrics["error_times"] = [e["time"] for e in errors]
            metrics["error_L2_values"] = [e["rel_L2"] for e in errors]

    # --- Probe CSV (point value at domain centre) ---
    probe_path = run_dir / "probe.csv"
    if probe_path.exists():
        with probe_path.open() as f:
            reader = csv.DictReader(f)
            probes = []
            for row in reader:
                probes.append(
                    {
                        "time": float(row["time"]),
                        "u_probe": float(row["u_probe"]),
                    }
                )

        if probes:
            metrics["probe_times"] = [p["time"] for p in probes]
            metrics["probe_values"] = [p["u_probe"] for p in probes]

    return metrics


def clean_double(x: float, precision: int = 6) -> str:
    """Replicate the C++ clean_double for matching folder names."""
    s = f"{x:.{precision}f}"
    s = s.rstrip("0").rstrip(".")
    return s.replace(".", "_") if s else "0"


def predict_run_folder(nel: int, r: int, dt: float, T: float, scheme_name: str) -> str:
    """Predict the run subfolder name generated by the C++ binary.

    Format: run-R{r}-N{nel}x{nel}-dt{dt_clean}-T{T_clean}{method_params}/
    """
    dt_clean = clean_double(dt)
    T_clean = clean_double(T)

    sdef = SCHEME_DEFS[scheme_name]
    if "Theta" in sdef["overrides"]:
        theta_clean = clean_double(float(sdef["overrides"]["Theta"]))
        method_params = f"-theta{theta_clean}"
    else:
        beta_clean = clean_double(float(sdef["overrides"]["Beta"]))
        gamma_clean = clean_double(float(sdef["overrides"]["Gamma"]))
        method_params = f"-gamma{gamma_clean}-beta{beta_clean}"

    return f"run-R{r}-N{nel}x{nel}-dt{dt_clean}-T{T_clean}{method_params}"


# ============================================================
# Main sweep
# ============================================================
def main():
    base = load_base_params()

    for s in args.schemes:
        if s not in SCHEME_DEFS:
            print(f"Unknown scheme: {s}. Available: {list(SCHEME_DEFS.keys())}")
            sys.exit(1)

    # Prepare directories
    logs_dir = Path.cwd() / "dissdisp-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    nel = args.nel
    nel_explicit = args.nel_explicit
    r_val = args.r

    # Build the run plan — explicit schemes use a coarser mesh to relax the CFL constraint,
    # ensuring theta=0 and Newmark CD produce stable (non-diverging) results.
    plan = []
    for scheme_name in args.schemes:
        nel_for_scheme = nel_explicit if SCHEME_DEFS[scheme_name]["explicit"] else nel
        for dt in sorted(args.dt, reverse=True):
            if is_cfl_safe(scheme_name, nel_for_scheme, r_val, dt):
                plan.append((scheme_name, dt, nel_for_scheme))
            else:
                cfl = cfl_limit(nel_for_scheme, r_val)
                print(f"  [SKIP] {scheme_name} dt={dt} exceeds CFL limit {cfl:.6f}")

    total = len(plan)
    print(f"\n{'='*60}")
    print(f"Dissipation/Dispersion sweep: {total} runs")
    print(f"  Schemes: {args.schemes}")
    print(f"  Nel (implicit): {nel}")
    print(f"  Nel (explicit): {nel_explicit}  (theta=0, Newmark CD)")
    print(f"  R:              {r_val}")
    print(f"  dt:             {args.dt}")
    print(f"  T:       {args.T}")
    print(f"  nprocs:  {args.nprocs}")
    print(f"  timeout: {args.timeout}s per run")
    print(f"{'='*60}\n")

    # Results collection
    all_metrics = []
    job_suffix = f"-{args.job_id}" if args.job_id else ""

    runlog_path = Path(f"dissdisp-runlog{job_suffix}.csv")
    with runlog_path.open("w") as logf:
        logf.write(
            "scheme,Nel,R,dt,T,returncode,elapsed_s,cfl_limit,"
            "energy_ratio,energy_decay_rate,max_rel_L2,final_rel_L2,final_rel_H1\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            param_file = Path(tmpdir) / f"{PARAM_STEM}.json"

            for i, (scheme_name, dt, nel_for_scheme) in enumerate(plan, 1):
                sdef = SCHEME_DEFS[scheme_name]
                cfl = (
                    cfl_limit(nel_for_scheme, r_val)
                    if sdef["explicit"]
                    else float("inf")
                )

                tag = f"{scheme_name}_Nel{nel_for_scheme}_R{r_val}_dt{dt}"
                print(
                    f"[{i}/{total}] {tag}"
                    + (f"  (CFL={cfl:.6f})" if sdef["explicit"] else "")
                )

                write_param_file(
                    base,
                    nel_for_scheme,
                    r_val,
                    dt,
                    args.T,
                    sdef["overrides"],
                    param_file,
                )

                code, elapsed = run_single(
                    sdef["binary"], param_file, tag, logs_dir, nel_for_scheme
                )
                status = (
                    "OK"
                    if code == 0
                    else ("TIMEOUT" if code == -1 else f"FAIL({code})")
                )
                print(f"  -> {status} in {elapsed:.1f}s")

                # Extract metrics from the run output
                results_base = Path("../results")
                prefix = "theta" if "theta" in scheme_name else "newmark"
                problem_name = f"{prefix}-{PARAM_STEM}"
                run_folder = predict_run_folder(
                    nel_for_scheme, r_val, dt, args.T, scheme_name
                )

                metrics = {}
                if code == 0:
                    metrics = extract_metrics(results_base, problem_name, run_folder)
                    if metrics:
                        print(
                            f"     Energy ratio E(T)/E(0) = {metrics.get('energy_ratio', 'N/A'):.8f}"
                        )
                        print(
                            f"     Final rel L2 error     = {metrics.get('final_rel_L2_error', 'N/A'):.6e}"
                        )

                logf.write(
                    f"{scheme_name},{nel_for_scheme},{r_val},{dt},{args.T},{code},{elapsed:.3f},{cfl:.8f},"
                    f"{metrics.get('energy_ratio', '')},{metrics.get('energy_decay_rate', '')},"
                    f"{metrics.get('max_rel_L2', '')},{metrics.get('final_rel_L2_error', '')},"
                    f"{metrics.get('final_rel_H1_error', '')}\n"
                )
                logf.flush()

                all_metrics.append(
                    {
                        "scheme": scheme_name,
                        "nel": nel_for_scheme,
                        "r": r_val,
                        "dt": dt,
                        "T": args.T,
                        **metrics,
                    }
                )

    # Write consolidated results CSV (compact, for the analysis notebook)
    summary_path = Path(f"dissdisp-results{job_suffix}.csv")
    with summary_path.open("w") as f:
        f.write(
            "scheme,Nel,R,dt,T,energy_ratio,energy_decay_rate,"
            "max_rel_L2,final_rel_L2,final_rel_H1\n"
        )
        for m in all_metrics:
            f.write(
                f"{m['scheme']},{m['nel']},{m['r']},{m['dt']},{m['T']},"
                f"{m.get('energy_ratio', '')},{m.get('energy_decay_rate', '')},"
                f"{m.get('max_rel_L2_error', '')},{m.get('final_rel_L2_error', '')},"
                f"{m.get('final_rel_H1_error', '')}\n"
            )

    # Also save the per-run energy time-series as individual CSVs
    # (for detailed time-domain analysis in the notebook)
    energy_dir = Path(f"dissdisp-energy-series{job_suffix}")
    energy_dir.mkdir(parents=True, exist_ok=True)
    for m in all_metrics:
        if "energy_times" in m:
            tag = f"{m['scheme']}_dt{m['dt']}"
            fpath = energy_dir / f"{tag}.csv"
            with fpath.open("w") as f:
                f.write("time,energy\n")
                for t_val, e_val in zip(m["energy_times"], m["energy_values"]):
                    f.write(f"{t_val},{e_val}\n")

    # Save per-run error time-series
    error_dir = Path(f"dissdisp-error-series{job_suffix}")
    error_dir.mkdir(parents=True, exist_ok=True)
    for m in all_metrics:
        if "error_times" in m:
            tag = f"{m['scheme']}_dt{m['dt']}"
            fpath = error_dir / f"{tag}.csv"
            with fpath.open("w") as f:
                f.write("time,rel_L2_error\n")
                for t_val, e_val in zip(m["error_times"], m["error_L2_values"]):
                    f.write(f"{t_val},{e_val}\n")

    # Save per-run probe time-series (point value at domain centre)
    probe_dir = Path(f"dissdisp-probe-series{job_suffix}")
    probe_dir.mkdir(parents=True, exist_ok=True)
    for m in all_metrics:
        if "probe_times" in m:
            tag = f"{m['scheme']}_dt{m['dt']}"
            fpath = probe_dir / f"{tag}.csv"
            with fpath.open("w") as f:
                f.write("time,u_probe\n")
                for t_val, p_val in zip(m["probe_times"], m["probe_values"]):
                    f.write(f"{t_val},{p_val}\n")

    print(f"\n{'='*60}")
    print(f"Done. Summary results: {summary_path}")
    print(f"Run log: {runlog_path}")
    print(f"Energy time-series: {energy_dir}/")
    print(f"Error time-series:  {error_dir}/")
    print(f"Probe time-series:  {probe_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
