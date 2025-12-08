import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convergence sweeps for wave solvers."
    )
    parser.add_argument(
        "--base-param",
        type=Path,
        default=Path("../parameters/sine-membrane.json"),
        help="Path to the base parameter JSON.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("../build/main-newmark"),
        help="Solver executable to run (e.g., ../build/main-newmark or ../build/main-theta).",
    )
    parser.add_argument(
        "--nel",
        nargs="+",
        default=["180,60", "120,40"],
        help="List of Nel pairs as 'nx,ny' strings.",
    )
    parser.add_argument(
        "--r", nargs="+", default=["1", "2"], help="List of polynomial degrees R."
    )
    parser.add_argument(
        "--dt", nargs="+", default=["0.015625", "0.01"], help="List of time steps Dt."
    )
    return parser.parse_args()


def load_base(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_temp_params(base: dict, nel: str, r: str, dt: str) -> Path:
    params = dict(base)
    params["Nel"] = nel.replace(" ", "")
    params["R"] = r
    params["Dt"] = dt
    tmp = tempfile.NamedTemporaryFile(
        prefix="convergence-", suffix=".json", delete=False
    )
    with tmp as f:
        json.dump(params, f, indent=2)
    return Path(tmp.name)


def run_case(binary: Path, param_file: Path) -> int:
    print(f"[RUN] {binary} with {param_file.name}")
    result = subprocess.run([str(binary), str(param_file)])
    return result.returncode


def main() -> None:
    args = parse_args()
    base = load_base(args.base_param)
    combos = itertools.product(args.nel, args.r, args.dt)
    for nel, r, dt in combos:
        param_file = write_temp_params(base, nel, r, dt)
        print(f"  Nel={nel}, R={r}, Dt={dt} -> {param_file}")
        try:
            code = run_case(args.binary, param_file)
            if code != 0:
                print(f"  [FAIL] Exit code {code}")
                break
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            break


if __name__ == "__main__":
    main()
