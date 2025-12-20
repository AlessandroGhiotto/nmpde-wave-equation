import itertools
import json
import subprocess
import tempfile
import time
import csv
from pathlib import Path
import os


# Manual configuration for convergence sweep
BASE_PARAM = Path("../parameters/standing-mode-wsol.json")


# Detect available CPUs from PBS or use fallback
def get_available_cpus():
    """Get number of CPUs from PBS_NODEFILE or ncpus environment variable."""
    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if pbs_nodefile and Path(pbs_nodefile).exists():
        with open(pbs_nodefile) as f:
            return len(f.readlines())

    ncpus = os.environ.get("PBS_NCPUS") or os.environ.get("SLURM_CPUS_ON_NODE")
    if ncpus:
        return int(ncpus)

    return 4  # fallback for local testing


MAX_CPUS = get_available_cpus()
# Only test up to available CPUs
MPI_PROCS = [n for n in [1, 2, 4, 8, 16] if n <= MAX_CPUS]

NEL_VALUES = ["120"]
R_VALUES = ["1"]
DT_VALUES = ["0.005"]
T_VALUE = "5.0"

BINARY_THETA = Path("../build/main-theta")
THETA_VALUE = "0.5"  # CN

BINARY_NEWMARK = Path("../build/main-newmark")
GAMMA_VALUE = "0.5"
BETA_VALUE = "0.25"

RESULTS_CSV = Path(__file__).with_name("scalability-results.csv")

# Timeout in seconds (adjust based on expected runtime)
TIMEOUT_SECONDS = 600  # 10 minutes per run


def load_base(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_temp_params(
    base: dict, nel: str, r: str, dt: str, overrides: dict, out_path: Path
) -> Path:
    params = dict(base)
    params["Nel"] = str(nel).replace(" ", "")
    params["R"] = str(r)
    params["Dt"] = str(dt)
    params["T"] = T_VALUE

    # apply scheme-specific parameters (e.g., Theta or Beta/Gamma)
    params.update(overrides)

    out_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return out_path


def run_case(binary: Path, param_file: Path, nprocs: int) -> tuple[int, float]:
    env = dict(**os.environ)

    # 1. Force single-threaded libraries
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # 2. MPI binding and mapping (critical on clusters)
    cmd = [
        "mpirun",
        "--bind-to",
        "core",
        "--map-by",
        "core",
        # REMOVED --oversubscribe - causes hanging on clusters
        "-np",
        str(nprocs),
        str(binary),
        str(param_file),
    ]

    # Add hostfile if available (PBS clusters)
    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if pbs_nodefile and Path(pbs_nodefile).exists():
        cmd.insert(1, "--hostfile")
        cmd.insert(2, pbs_nodefile)

    print(f"[RUN] {' '.join(cmd)}")
    print(f"[INFO] Available CPUs: {MAX_CPUS}, Using: {nprocs}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=TIMEOUT_SECONDS,
            capture_output=False,  # Let output go to stdout/stderr
        )
        t1 = time.perf_counter()
        return result.returncode, (t1 - t0)
    except subprocess.TimeoutExpired:
        t1 = time.perf_counter()
        print(f"[TIMEOUT] Run exceeded {TIMEOUT_SECONDS}s")
        return -999, (t1 - t0)


def main() -> None:
    base = load_base(BASE_PARAM)

    schemes = [
        ("theta", BINARY_THETA, {"Theta": THETA_VALUE}),
        ("newmark", BINARY_NEWMARK, {"Beta": BETA_VALUE, "Gamma": GAMMA_VALUE}),
    ]

    combos = itertools.product(NEL_VALUES, R_VALUES, DT_VALUES)

    header = [
        "scheme",
        "binary",
        "nprocs",
        "Nel",
        "R",
        "Dt",
        "T",
        "Theta",
        "Beta",
        "Gamma",
        "returncode",
        "seconds",
    ]

    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open(
        "a", newline="", encoding="utf-8"
    ) as fcsv, tempfile.TemporaryDirectory() as tmpdir:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        if write_header:
            writer.writeheader()

        for scheme_name, binary, overrides in schemes:
            if not binary.is_file():
                print(f"[SKIP] Missing binary: {binary}")
                continue

            for nprocs in MPI_PROCS:
                for nel, r, dt in itertools.product(NEL_VALUES, R_VALUES, DT_VALUES):
                    tmp_param = (
                        Path(tmpdir)
                        / f"params_{scheme_name}_p{nprocs}_nel{nel}_r{r}_dt{dt}.json"
                    )
                    param_file = write_temp_params(
                        base, nel, r, dt, overrides, tmp_param
                    )

                    print("=" * 40)
                    print(
                        f"  scheme={scheme_name}, nprocs={nprocs}, Nel={nel}, R={r}, Dt={dt}"
                    )

                    try:
                        code, seconds = run_case(binary, param_file, nprocs)
                    except Exception as exc:
                        code, seconds = -1, 0.0
                        print(f"  [ERROR] {exc}")

                    row = {
                        "scheme": scheme_name,
                        "binary": str(binary),
                        "nprocs": nprocs,
                        "Nel": nel,
                        "R": r,
                        "Dt": dt,
                        "T": T_VALUE,
                        "Theta": overrides.get("Theta", ""),
                        "Beta": overrides.get("Beta", ""),
                        "Gamma": overrides.get("Gamma", ""),
                        "returncode": code,
                        "seconds": f"{seconds:.6f}",
                    }
                    writer.writerow(row)
                    fcsv.flush()

                    if code != 0:
                        print(f"  [FAIL] Exit code {code} (continuing)")


if __name__ == "__main__":
    main()
