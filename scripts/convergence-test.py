import itertools
import json
import subprocess
import tempfile
from pathlib import Path


# Manual configuration for convergence sweep
BASE_PARAM = Path("../parameters/standing-mode-wsol.json")
BINARY = Path("../build/main-theta")
MPI_PROCS = 4
NEL_VALUES = ["20", "40", "80", "160", "320"]
R_VALUES = ["1", "2"]
DT_VALUES = ["0.05", "0.01", "0.005"]


def load_base(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_temp_params(base: dict, nel: str, r: str, dt: str) -> Path:
    params = dict(base)
    params["Nel"] = nel.replace(" ", "")
    params["R"] = r
    params["Dt"] = dt
    with BASE_PARAM.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return BASE_PARAM


def run_case(binary: Path, param_file: Path) -> int:
    print(f"[RUN] mpirun -n {MPI_PROCS} {binary} with {param_file.name}")
    result = subprocess.run(
        ["mpirun", "-n", str(MPI_PROCS), str(binary), str(param_file)]
    )
    return result.returncode


def main() -> None:
    base = load_base(BASE_PARAM)
    original_content = BASE_PARAM.read_text(encoding="utf-8")
    combos = itertools.product(NEL_VALUES, R_VALUES, DT_VALUES)
    try:
        for nel, r, dt in combos:
            param_file = write_temp_params(base, nel, r, dt)
            print(f"  Nel={nel}, R={r}, Dt={dt} -> {param_file}")
            try:
                code = run_case(BINARY, param_file)
                if code != 0:
                    print(f"  [FAIL] Exit code {code}")
                    break
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                break
    finally:
        BASE_PARAM.write_text(original_content, encoding="utf-8")


if __name__ == "__main__":
    main()
