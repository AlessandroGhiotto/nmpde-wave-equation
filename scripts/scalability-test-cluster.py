import itertools
import json
import subprocess
import tempfile
import time
import csv
from pathlib import Path
import os
import shlex

# Make all paths robust to current working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../wave-equation

# Manual configuration for convergence sweep
BASE_PARAM = PROJECT_DIR / "parameters" / "standing-mode-wsol.json"
MPI_PROCS = [1, 2, 4, 8, 16]

NEL_VALUES = ["120"]
R_VALUES = ["1"]
DT_VALUES = ["0.005"]
T_VALUE = "5.0"

BINARY_THETA = PROJECT_DIR / "build" / "main-theta"
THETA_VALUE = "0.5"  # CN

BINARY_NEWMARK = PROJECT_DIR / "build" / "main-newmark"
GAMMA_VALUE = "0.5"
BETA_VALUE = "0.25"

RESULTS_CSV = Path(__file__).with_name("scalability-results.csv")
LOG_DIR = Path(__file__).with_name("scalability-logs")

# Environment-configurable launcher behavior (helps on clusters)
MPI_LAUNCHER = os.environ.get("MPI_LAUNCHER", "mpirun")  # e.g. mpiexec, mpirun, srun
MPI_EXTRA_ARGS = shlex.split(
    os.environ.get("MPI_EXTRA_ARGS", "")
)  # e.g. "--report-bindings"
CASE_TIMEOUT_SECONDS = float(os.environ.get("CASE_TIMEOUT_SECONDS", "0"))  # 0 disables


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


def _pbs_hostfile_args() -> list[str]:
    pbs_nodefile = os.environ.get("PBS_NODEFILE", "")
    if pbs_nodefile and Path(pbs_nodefile).is_file():
        # OpenMPI/MPICH accept different flags; --hostfile is common for OpenMPI.
        # If your site uses a different launcher, override via MPI_EXTRA_ARGS/MPI_LAUNCHER.
        return ["--hostfile", pbs_nodefile]
    return []


def _mpi_cmd(nprocs: int, binary: Path, param_file: Path) -> list[str]:
    # Avoid --oversubscribe by default on allocated nodes (can cause bad behavior on some MPIs).
    base = [MPI_LAUNCHER, "-np", str(nprocs)]
    base += _pbs_hostfile_args()
    base += MPI_EXTRA_ARGS
    return base + [str(binary), str(param_file)]


def _pbs_slots() -> int | None:
    pbs_nodefile = os.environ.get("PBS_NODEFILE", "")
    if pbs_nodefile and Path(pbs_nodefile).is_file():
        try:
            return sum(1 for _ in Path(pbs_nodefile).read_text(encoding="utf-8").splitlines() if _.strip())
        except Exception:
            return None
    return None


def run_case(
    binary: Path, param_file: Path, nprocs: int, log_path: Path
) -> tuple[int, float]:
    env = dict(**os.environ)

    # 1. Force single-threaded libraries
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = _mpi_cmd(nprocs, binary, param_file)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {' '.join(cmd)}")
    print(f"[LOG] {log_path}")

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as flog:
        # Provenance: helps catch host-mpirun vs container-libmpi mismatch immediately
        flog.write(f"CMD: {' '.join(cmd)}\n")
        flog.write(f"CWD: {str(binary.parent)}\n")
        flog.write(f"MPI_LAUNCHER: {MPI_LAUNCHER}\n")
        flog.write(f"PBS_NODEFILE: {os.environ.get('PBS_NODEFILE','')}\n")
        slots = _pbs_slots()
        flog.write(f"PBS_SLOTS(from nodefile): {slots}\n")
        flog.flush()

        if slots is not None and nprocs > slots:
            flog.write(f"ERROR: requested -np {nprocs} but PBS_NODEFILE advertises {slots} slots\n")
            flog.flush()
            return -2, (time.perf_counter() - t0)

        # Best-effort diagnostics (don’t fail the run if these commands fail)
        try:
            subprocess.run(["bash", "-lc", "which mpirun && mpirun --version"], stdout=flog, stderr=subprocess.STDOUT, text=True)
        except Exception:
            flog.write("WARN: failed to capture mpirun version\n")
        try:
            subprocess.run(["bash", "-lc", f"ldd {shlex.quote(str(binary))} | grep -i mpi || true"], stdout=flog, stderr=subprocess.STDOUT, text=True)
        except Exception:
            flog.write("WARN: failed to capture ldd/mpi linkage\n")
        flog.flush()

        # start_new_session=True lets us kill the whole mpirun process group on timeout/hang.
        p = subprocess.Popen(
            cmd,
            cwd=str(binary.parent),
            env=env,
            stdout=flog,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            p.wait(timeout=None if CASE_TIMEOUT_SECONDS <= 0 else CASE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Hard-kill the whole session (mpirun + ranks) to avoid orphaned MPI processes.
            try:
                os.killpg(p.pid, 9)
            except Exception:
                p.kill()
            return -9, (time.perf_counter() - t0)

    t1 = time.perf_counter()
    return int(p.returncode or 0), (t1 - t0)


def main() -> None:
    base = load_base(BASE_PARAM)

    schemes = [
        ("theta", BINARY_THETA, {"Theta": THETA_VALUE}),
        ("newmark", BINARY_NEWMARK, {"Beta": BETA_VALUE, "Gamma": GAMMA_VALUE}),
    ]

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

                    log_path = (
                        LOG_DIR
                        / f"log_{scheme_name}_p{nprocs}_nel{nel}_r{r}_dt{dt}.txt"
                    )

                    print("=" * 40)
                    print(
                        f"  scheme={scheme_name}, nprocs={nprocs}, Nel={nel}, R={r}, Dt={dt}"
                    )

                    try:
                        code, seconds = run_case(binary, param_file, nprocs, log_path)
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
                        print(f"         See log: {log_path}")


if __name__ == "__main__":
    main()
