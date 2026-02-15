import argparse
import json
import subprocess
import tempfile
import time
import csv
import os
from pathlib import Path

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--nprocs", type=int, required=True)
parser.add_argument(
    "--repeats", type=int, default=1, help="Number of repetitions per scheme."
)
parser.add_argument(
    "--launcher",
    default=os.environ.get("MPI_LAUNCHER", "mpirun"),
    help="MPI launcher command (e.g. mpirun, mpiexec). Default: $MPI_LAUNCHER or mpirun",
)
parser.add_argument(
    "--mpi-arg",
    action="append",
    default=[],
    help="Extra MPI launcher arg (repeatable). Example: --mpi-arg=--bind-to --mpi-arg=core",
)
parser.add_argument(
    "--bind-to-core",
    action="store_true",
    default=True,
    help="Add --bind-to core --map-by core to mpirun (default: True). "
    "Ensures each MPI rank lands on its own physical core. "
    "Disable with --no-bind-to-core if your MPI does not support it.",
)
parser.add_argument(
    "--no-bind-to-core",
    action="store_false",
    dest="bind_to_core",
)
parser.add_argument(
    "--use-pbs-nodefile",
    action="store_true",
    help="If $PBS_NODEFILE is set, pass it as a hostfile to the launcher (OpenMPI-style --hostfile).",
)
args = parser.parse_args()

NPROCS = args.nprocs

# ----------------------------
# Configuration (robust paths)
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_PARAM = (PROJECT_ROOT / "parameters" / "standing-mode-wsol.json").resolve()

NEL = "320"
R = "1"
DT = "0.005"
T_VALUE = "3.0"

BINARY_THETA = (PROJECT_ROOT / "build" / "main-theta").resolve()
BINARY_NEWMARK = (PROJECT_ROOT / "build" / "main-newmark").resolve()

THETA_VALUE = "0.5"
BETA_VALUE = "0.25"
GAMMA_VALUE = "0.5"

RESULTS_CSV = Path(f"scalability-results-{NPROCS}.csv")


# ----------------------------
def load_base(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_params(base, overrides, out_path):
    params = dict(base)
    params.update(
        {
            "Nel": NEL,
            "R": R,
            "Dt": DT,
            "T": T_VALUE,
            "Print Every": "10000000",
            "Log Every": "0",
            "Enable Logging": "false",
            "Save Solution": "false",
        }
    )
    params.update(overrides)
    out_path.write_text(json.dumps(params, indent=2))
    return out_path


def _build_mpi_cmd(binary: Path, param_file: Path) -> list[str]:
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    cmd = [args.launcher, "-np", str(NPROCS)]

    # Optional: integrate PBS nodefile (common on clusters).
    # We generate a clean hostfile with exactly NPROCS slots on the first
    # hostname found in $PBS_NODEFILE.  This avoids scattering ranks across
    # entries when the original nodefile has more lines than NPROCS.
    if args.use_pbs_nodefile:
        pbs_nodefile = os.environ.get("PBS_NODEFILE")
        if pbs_nodefile and Path(pbs_nodefile).exists():
            # Read unique hostname(s); use the first one (single-node job).
            hosts = list(dict.fromkeys(Path(pbs_nodefile).read_text().splitlines()))
            host = hosts[0] if hosts else None
            if host:
                clean_hf = Path(f"/tmp/hostfile_np{NPROCS}_{os.getpid()}")
                clean_hf.write_text(f"{host} slots={NPROCS}\n")
                cmd += ["--hostfile", str(clean_hf)]

    # Explicit core binding: ensures each MPI rank gets its own physical core.
    # Prevents accidental hyperthreading (2 ranks sharing 1 core).
    if args.bind_to_core and NPROCS > 1:
        cmd += ["--bind-to", "core", "--map-by", "core"]

    # User-provided MPI args (portable way to encode site-specific binding/mapping).
    cmd += list(args.mpi_arg)

    cmd += [str(binary), str(param_file)]
    return cmd


def run(binary: Path, param_file: Path, *, log_dir: Path, run_tag: str):
    cmd = _build_mpi_cmd(binary, param_file)

    stdout_path = log_dir / f"{run_tag}.out"
    stderr_path = log_dir / f"{run_tag}.err"

    print("[RUN]", " ".join(cmd))
    t0 = time.perf_counter()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        r = subprocess.run(cmd, stdout=out, stderr=err)
    t1 = time.perf_counter()

    return r.returncode, t1 - t0, " ".join(cmd), str(stdout_path), str(stderr_path)


# ----------------------------
def main():
    base = load_base(BASE_PARAM)

    schemes = [
        ("theta", BINARY_THETA, {"Theta": THETA_VALUE}),
        ("newmark", BINARY_NEWMARK, {"Beta": BETA_VALUE, "Gamma": GAMMA_VALUE}),
    ]

    header = [
        "scheme",
        "binary",
        "nprocs",
        "repeat",
        "Nel",
        "R",
        "Dt",
        "T",
        "Theta",
        "Beta",
        "Gamma",
        "returncode",
        "seconds",
        "cmd",
        "stdout_log",
        "stderr_log",
    ]

    with RESULTS_CSV.open("w", newline="") as f, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logs_dir = Path.cwd() / "run-logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for name, binary, overrides in schemes:
            param_file = tmp_path / f"params_{name}.json"
            write_params(base, overrides, param_file)

            for rep in range(1, args.repeats + 1):
                tag = f"{name}_p{NPROCS}_r{rep}"
                code, secs, cmd, out_log, err_log = run(
                    binary, param_file, log_dir=logs_dir, run_tag=tag
                )

                writer.writerow(
                    {
                        "scheme": name,
                        "binary": str(binary),
                        "nprocs": NPROCS,
                        "repeat": rep,
                        "Nel": NEL,
                        "R": R,
                        "Dt": DT,
                        "T": T_VALUE,
                        "Theta": overrides.get("Theta", ""),
                        "Beta": overrides.get("Beta", ""),
                        "Gamma": overrides.get("Gamma", ""),
                        "returncode": code,
                        "seconds": f"{secs:.6f}",
                        "cmd": cmd,
                        "stdout_log": out_log,
                        "stderr_log": err_log,
                    }
                )


if __name__ == "__main__":
    main()
