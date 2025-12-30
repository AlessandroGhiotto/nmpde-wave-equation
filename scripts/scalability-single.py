import argparse
import json
import subprocess
import tempfile
import time
import csv
from pathlib import Path

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--nprocs", type=int, required=True)
args = parser.parse_args()

NPROCS = args.nprocs

# ----------------------------
# Configuration
# ----------------------------
BASE_PARAM = Path("../parameters/standing-mode-wsol.json")

NEL = "10"
R = "1"
DT = "0.005"
T_VALUE = "3.0"

BINARY_THETA = Path("../build/main-theta")
BINARY_NEWMARK = Path("../build/main-newmark")

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
            "Print Every": "200",
        }
    )
    params.update(overrides)
    out_path.write_text(json.dumps(params, indent=2))
    return out_path


def run(binary, param_file):
    cmd = [
        "mpirun",
        "-np",
        str(NPROCS),
        "--bind-to",
        "core",
        "--map-by",
        "ppr:1:core",
        str(binary),
        str(param_file),
    ]
    print("[RUN]", " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    t1 = time.perf_counter()
    return r.returncode, t1 - t0


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

    with RESULTS_CSV.open("w", newline="") as f, tempfile.TemporaryDirectory() as tmp:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for name, binary, overrides in schemes:
            param_file = Path(tmp) / f"params_{name}.json"
            write_params(base, overrides, param_file)

            code, secs = run(binary, param_file)

            writer.writerow(
                {
                    "scheme": name,
                    "binary": str(binary),
                    "nprocs": NPROCS,
                    "Nel": NEL,
                    "R": R,
                    "Dt": DT,
                    "T": T_VALUE,
                    "Theta": overrides.get("Theta", ""),
                    "Beta": overrides.get("Beta", ""),
                    "Gamma": overrides.get("Gamma", ""),
                    "returncode": code,
                    "seconds": f"{secs:.6f}",
                }
            )


if __name__ == "__main__":
    main()
