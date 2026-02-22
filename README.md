# Wave Equation — FEM Solver (deal.II)

Parallel finite-element solver for the 2D scalar wave equation

$$
\left\lbrace
\begin{array}{lll}
\displaystyle \frac{\partial^2 u}{\partial t^2} - c^2 \Delta u = f
& \text{in } \Omega \times [0,T]
& {} \\\\[10pt]
u(x,t) = g
& \text{on } \partial\Omega \times [0,T]
& \text{(Dirichlet B.C.)} \\\\[10pt]
u(x,0) = u_0(x)
& \text{in } \Omega
& \text{(Initial Displacement)} \\\\[10pt]
\displaystyle \frac{\partial u}{\partial t}(x,0) = v_0(x)
& \text{in } \Omega
& \text{(Initial Velocity)}
\end{array}
\right.
$$

on a rectangular domain $\Omega$ with simplicial (triangular) elements, built using [deal.II](https://www.dealii.org/), exploiting Trilinos and MPI
parallelism.


## Implemented Time-Integration Families

<div align="center">
  
| Method           |     Parameters     | Key properties                                                                                                        |
| :--------------- | :----------------: | :-------------------------------------------------------------------------------------------------------------------- |
| **Theta-method** | $\theta \in [0,1]$ | $\theta=0$ → Forward Euler (FE) <br/> $\theta=\tfrac12$ → Crank–Nicolson (CN) <br/> $\theta=1$ →  Backward Euler (BE) |
| **Newmark-β**    |   $\gamma,\beta$   | $\beta=0$ → Central Difference method <br/>$\beta=\tfrac14, \gamma=\tfrac12$ → Middle Point rule                      |

</div>

---

## Repository structure

```
nmpde-wave-equation/
├── include/              # C++ headers
│   ├── WaveEquationBase.hpp   # abstract base (mesh, FE, logging, output)
│   ├── WaveNewmark.hpp        # Newmark-beta solver
│   ├── WaveTheta.hpp          # theta-method solver
│   └── ParameterReader.hpp    # JSON/PRM parameter parser
├── src/                  # C++ sources
│   ├── WaveEquationBase.cpp
│   ├── WaveNewmark.cpp
│   ├── WaveTheta.cpp
│   ├── ParameterReader.cpp
│   ├── main-newmark.cpp       # executable entry point (Newmark)
│   └── main-theta.cpp         # executable entry point (theta)
├── parameters/           # JSON parameter files for different test cases
├── scripts/              # Python sweep scripts + PBS job files
├── analysis/             # Jupyter notebooks for post-processing
├── results/              # simulation output (VTU, CSV) — git-ignored
├── build/                # CMake build directory — git-ignored
└── report/               # LaTeX report
```

---

## Building

### Prerequisites

- C++17 compiler with MPI support
- [deal.II](https://www.dealii.org/) ≥ 9.3.1 (with Trilinos enabled)

You can get the Apptainer containter used in the project by executing the following commands on a terminal window

```bash
mkdir -p $HOME/apptainer-tmp/
mkdir -p $HOME/apptainer-cache/
export APPTAINER_TMPDIR=$HOME/apptainer-tmp/
export APPTAINER_CACHEDIR=$HOME/apptainer-cache/
apptainer pull docker://quay.io/pjbaioni/amsc_mk:2025
```

and afterwards load the container and the needed modules

```bash
apptainer shell /path/to/amsc_mk_2025.sif
source /u/sw/etc/bash.bashrc
module load gcc-glibc dealii
```

### Compile

```bash
mkdir build
cd build
cmake ..
make -j
```

This produces two executables: `main-theta` and `main-newmark`.

---

## Running a simulation

Both executables take a single command-line argument i.e. the path to a `JSON` parameter file.

```bash
cd build
mpirun -np 4 ./main-theta   ../parameters/standing-mode-wsol.json
mpirun -np 4 ./main-newmark ../parameters/gaussian-pulse.json
```

If no argument is given, the default `../parameters/sine-membrane.json` is used.

### Parameter file format

A `JSON` parameter file with scalar entries and function subsections

```jsonc
{
  "Geometry": "[0.0, 1.0] x [0.0, 1.0]",             // domain bounding box
  "Nel": "80",                                       // elements per side (or "80, 60" for rectangular)
  "R": "1",                                          // polynomial degree
  "T": "1.0",                                        // final time
  "Theta": "0.5",                                    // theta parameter (theta-method only)
  "Beta": "0.25",                                    // beta  parameter (Newmark only)
  "Gamma": "0.5",                                    // gamma parameter (Newmark only)
  "Dt": "0.005",                                     // time step
  "Save Solution": true,                             // write VTU output
  "Enable Logging": true,                            // write energy/error CSVs
  "Log Every": 10,                                   // log frequency (0 = off)
  "Print Every": 10,                                 // console output frequency
  "C":  { "Function expression": "1.0", ... },       // wave speed c(x,y,t)
  "F":  { "Function expression": "0.0", ... },       // forcing f(x,y,t)
  "U0": { "Function expression": "sin(pi*x)*sin(pi*y)", ... },  // initial displacement
  "V0": { "Function expression": "0.0", ... },       // initial velocity
  "G":  { "Function expression": "0.0", ... },       // Dirichlet BC for u
  "DGDT": { "Function expression": "0.0", ... },     // time-derivative of g
  "Solution": { "Function expression": "...", ... }  // exact solution (optional)
}
```

Several already available parameter files are provided in `parameters/` (e.g. Gaussian pulse, Ricker wavelet, ...).

### Output

The results of the simulation are written to the following folder `/results/<problem_name>/run-R...-N...-dt.../`:

- `solution_*.vtu` or `solution_*.pvtu` → displacement, velocity, and eventually the exact solution, ideal for being imported in ParaView
- `energy.csv` → discrete energy time series
- `error.csv` → L2 and H1 error and comparison w.r.t the exact solution, if available
- `probe.csv` → point probe at the domain centre
- `iterations.csv` → Conjugate Gradient (CG) method iteration counts
- `convergence.csv` → execution summary appended from different runs

---

## C++ code overview

### WaveEquationBase

Abstract base class providing:

- Mesh creation (`subdivided_hyper_rectangle_with_simplices`) and parallel partitioning
- FE space setup (`FE_SimplexP`, `QGaussSimplex`)
- DoF distribution
- Energy computation $E^n = \tfrac12 (\mathbf{v}^T M \mathbf{v} + \mathbf{u}^T K \mathbf{u})$
- L2 / H1 error integration against an exact solution
- VTU/PVTU output, CSV logging, divergence detection

### WaveTheta

Inherits from `WaveEquationBase`. Rewrites the wave equation as a first-order
system and applies the theta-method. Each time step solves **two** SPD systems
(one for $u^{n+1}$, one for $v^{n+1}$) with CG + AMG.

### WaveNewmark

Inherits from `WaveEquationBase`. Uses the Newmark-$\beta$ family. Each time
step solves **one** SPD system for the acceleration $a^{n+1}$, then updates
$u$ and $v$ algebraically. A consistent initial acceleration $a^0$ is computed
at startup by solving $M a^0 = f(0) - K u^0$.

### ParameterReader

Thin wrapper around deal.II's `ParameterHandler`. Declares scalar parameters
and function subsections, parses JSON/PRM files, and initialises `FunctionParser`
objects (supporting symbolic `pi` constants).

---

## Python scripts

All scripts live in `scripts/` and drive parametric studies by repeatedly
invoking the C++ executables with different parameter combinations.

| Python script                     | Purpose                                                                                                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `convergence_sweep.py`            | Sweep over `(scheme, Nel, R, dt)` to study spatial and temporal convergence, applying also Courant-Friedrichs-Lewy (CFL) filtering for explicit methods.     |
| `dissipation_dispersion_sweep.py` | Fix the mesh, sweep over `dt` for each scheme, logging energy values, errors and point-probing at each step to analyse numerical dissipation and dispersion. |
| `scalability_sweep.py`            | Fix discretisation, measure the wall-clock time for a given number of MPI processes.                                                                         |

### Common CLI flags

```bash
python3 convergence_sweep.py \
    --nprocs 4 \
    --nel 10 20 40 80 \
    --r 1 2 \
    --dt 0.01 0.005 0.001 \
    --T 1.0 \
    --schemes theta-0.5 newmark-0.25 \
    --timeout 600 \
    --cfl-safety 0.9
```

Each script produces CSV result files that are later read by the analysis notebooks.

---

## Running on the cluster with the PBS scheduler

Three PBS job scripts are provided in `scripts/`:

| PBS script                       | What it runs                                               |
| -------------------------------- | ---------------------------------------------------------- |
| `convergence_all.pbs`            | `convergence_sweep.py` with 16 MPI processes               |
| `dissipation_dispersion_all.pbs` | `dissipation_dispersion_sweep.py` with 16 MPI processes    |
| `scalability_all.pbs`            | `scalability_sweep.py` for p = 1, 2, 4, 8, 16 (sequential) |

You should be sure to change the directories used in the scripts before submitting the job, and put your desired one.

### Submitting a job

```bash
cd wave-equation/scripts
qsub convergence_all.pbs
```

### What the PBS jobs do

1. **Copy the project to `scratch_local`** — the `scripts/`, `parameters/`
   and `build/` directories are copied to `/scratch_local/nmpde-<name>_${PBS_JOBID}/`.
   All computation happens on this fast node-local temporary storage and
   get much better I/O performance for the many small files produced by the sweeps.
2. **Run the Python sweep script** from the scratch directory with `--use-pbs-nodefile`
   and `--bind-to-core` binding MPI processes to cores, not to threads
3. **Copy results back** to a persistent location under the user's home directory
   (CSVs, compressed logs, raw convergence files).

Other notes:

- `OMP_NUM_THREADS=1` is set to prevent OpenMP oversubscription.
- For scalability tests, select a free node, so that memory bandwidth is not shared with other jobs.
- Core binding is enabled via `--bind-to core --map-by socket`.

---

## Analysis notebooks

The `analysis/` folder contains Jupyter notebooks that read the `.csv` files produced by the Python scripts.

| Notebook                                | Content                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `convergence-analysis.ipynb`            | Convergence plots (relative L2/H1 error vs. $h$ and vs. $\Delta t$) for all five schemes. Computes observed convergence rates. |
| `dissipation-dispersion-analysis.ipynb` | Energy ratio $E(T)/E(0)$ vs. $\Delta t$ (dissipation) and point-probe time series vs. exact solution (dispersion).             |
| `scalability-analisys.ipynb`            | Strong-scaling plots: wall time, speedup and parallel efficiency vs. number of MPI processes. Includes Amdahl's law fit.       |

All the notebooks expect the `.csv` data to be placed in the folder `analysis/data/`.

---

## Acknowledgments

We would like to thank Prof. [Michele Bucelli](https://github.com/michelebucelli) for its valuable advices during the execution of the project.

---
