/**
 * @file WaveEquationBase.hpp
 * @brief Abstract base class for parallel finite-element wave-equation solvers.
 *
 * Provides the mesh setup, FE space initialisation, assembly infrastructure,
 * energy/error logging, VTU output, and MPI bookkeeping that are shared by
 * every time-stepping scheme (theta-method, Newmark, …).  Concrete solvers
 * inherit from WaveEquationBase and implement the pure-virtual run() method.
 *
 * Built on top of the deal.II library with Trilinos linear-algebra back-end.
 */

#ifndef WAVE_EQUATION_BASE_HPP
#define WAVE_EQUATION_BASE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace dealii;

/**
 * @class WaveEquationBase
 * @brief Abstract base class for 2-D wave-equation solvers.
 *
 * Solves the scalar wave equation
 * @f[
 *   \partial_{tt} u - c^2 \Delta u = f \quad \text{in } \Omega \times (0,T]
 * @f]
 * on a rectangular simplicial mesh distributed across MPI ranks.
 *
 * Derived classes (WaveTheta, WaveNewmark) implement the actual
 * time-stepping loop by overriding run().
 */
class WaveEquationBase
{
  public:
    /** @brief Spatial dimension (2-D). */
    static constexpr unsigned int dim = 2;

    /** @brief Virtual destructor for safe polymorphic deletion. */
    virtual ~WaveEquationBase() = default;

    /**
     * @brief Execute the full simulation (pure-virtual).
     *
     * Concrete solvers set up the mesh, assemble matrices,
     * march in time and produce output.
     */
    virtual void run() = 0;

  protected:
    /**
     * @brief Construct the base solver (called by derived-class constructors).
     *
     * @param problem_name_    Human-readable name used for output folder naming.
     * @param N_el_            Number of elements in x- and y-direction.
     * @param geometry_        Bounding box of the rectangular domain
     *                         (bottom-left, top-right corners).
     * @param r_               Polynomial degree of the FE space.
     * @param T_               Final simulation time.
     * @param delta_t_         Time-step size.
     * @param c_               Wave-speed function @f$ c(\mathbf x) @f$.
     * @param f_               Forcing (source) term @f$ f(\mathbf x,t) @f$.
     * @param u0_              Initial displacement @f$ u(\mathbf x,0) @f$.
     * @param v0_              Initial velocity @f$ \partial_t u(\mathbf x,0) @f$.
     * @param g_               Dirichlet boundary data for @f$ u @f$.
     * @param dgdt_            Time derivative of the Dirichlet data
     *                         @f$ \partial_t g @f$ (needed by the theta-method
     *                         for the velocity system).
     * @param log_every_       Write energy / error CSVs every n steps (0 = off).
     * @param print_every_     Print step info to stdout every n steps.
     * @param exact_solution_  Pointer to the manufactured exact solution
     *                         (nullptr when not available).
     */
    WaveEquationBase(
        const std::string& problem_name_,
        const std::pair<unsigned int, unsigned int>& N_el_,
        const std::pair<Point<dim>, Point<dim>>& geometry_,
        const unsigned int& r_,
        const double& T_,
        const double& delta_t_,
        const Function<dim>& c_,
        Function<dim>& f_,
        const Function<dim>& u0_,
        const Function<dim>& v0_,
        Function<dim>& g_,
        Function<dim>& dgdt_,
        const unsigned int log_every_,
        const unsigned int print_every_,
        Function<dim>* exact_solution_)
        : problem_name(problem_name_), N_el(N_el_), geometry(geometry_),
          r(r_), T(T_), delta_t(delta_t_), c(c_), f(f_),
          u0(u0_), v0(v0_), g(g_), dgdt(dgdt_),
          log_every(log_every_), print_every(print_every_),
          exact_solution(exact_solution_),
          mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
          mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
          mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0)
    {
    }

    // ---- Mesh & FE initialisation ----------------------------------------

    /** @brief Create a simplicial rectangle, partition it and build the
     *         distributed triangulation. Also writes a VTK file of the mesh. */
    void setup_mesh();

    /** @brief Instantiate the FE_SimplexP element and Gauss quadrature rule. */
    void setup_fe();

    /** @brief Distribute DoFs on the mesh. */
    void setup_dof_handler();

    // ---- Output & logging -------------------------------------------------

    /**
     * @brief Build the output folder path and open convergence CSV.
     * @param method_params  Suffix encoding method-specific parameters
     *                       (e.g. "-theta0_5" or "-gamma0_5-beta0_25").
     */
    void prepare_output_filename(const std::string& method_params);

    /** @brief Compute the discrete energy
     *         @f$ E^n = \tfrac12 (v^T M v + u^T K u) @f$
     *         and append it to the energy CSV log. */
    void compute_and_log_energy();

    /** @brief Evaluate @f$ u @f$ at the centre of the domain and log to CSV. */
    void log_point_probe();

    /**
     * @brief Log the number of CG iterations for the current time step.
     * @param n_iterations_1  Iteration count for the first linear system.
     * @param n_iterations_2  Iteration count for the second system (0 if N/A).
     */
    void log_iterations(const unsigned int n_iterations_1, const unsigned int n_iterations_2);

    /** @brief Compute L2 and H1 errors against the exact solution and log to CSV. */
    void compute_and_log_error();

    /** @brief Compute and log final errors (convenience overload with empty strings). */
    void compute_final_errors();

    /**
     * @brief Compute final errors and append a row to the convergence CSV.
     * @param theta_str  String representation of theta (or empty).
     * @param beta_str   String representation of beta  (or empty).
     * @param gamma_str  String representation of gamma (or empty).
     */
    void compute_final_errors(const std::string& theta_str,
                              const std::string& beta_str,
                              const std::string& gamma_str);

    /** @brief Print a one-line summary of the current time step to stdout. */
    void print_step_info();

    /** @brief Write VTU/PVTU output files for the current time step. */
    void output() const;

    // ---- Error computation ------------------------------------------------

    /**
     * @brief Integrate the error between the FE solution and an exact function.
     * @param cell_norm       Norm type (L2_norm, H1_norm, …).
     * @param exact_solution  Reference exact-solution function.
     * @return Global (reduced across MPI ranks) error value.
     */
    double compute_error(const VectorTools::NormType& cell_norm, const Function<dim>& exact_solution) const;

    /**
     * @brief Compute a relative error  @f$ \|u_h - u\| / \|u\| @f$.
     * @param error      Absolute error (previously computed).
     * @param norm_type  Norm type used for the denominator.
     * @param exact_solution  Exact-solution function.
     * @return Relative error (returns absolute error when the exact norm is < 1e-14).
     */
    double compute_relative_error(const double error,
                                  const VectorTools::NormType& norm_type,
                                  const Function<dim>& exact_solution) const;

    /**
     * @brief Check whether the solution norms exceed a threshold (divergence detector).
     * @param norm_u     L2 norm of @f$ u @f$.
     * @param norm_v     L2 norm of @f$ v @f$.
     * @param threshold  Blow-up threshold.
     * @return True if the solution has diverged.
     */
    bool check_divergence(const double norm_u,
                          const double norm_v,
                          const double threshold) const;

    // ---- Problem description ----------------------------------------------

    /** @brief Human-readable problem name (used in folder paths). */
    const std::string problem_name;

    /** @brief Full path of the output folder for this run. */
    std::string output_folder;

    /** @brief Output stream for the energy time-series CSV. */
    std::ofstream energy_log_file;

    /** @brief Output stream for the error time-series CSV. */
    std::ofstream error_log_file;

    /** @brief Output stream for the convergence-study CSV (appended across runs). */
    std::ofstream convergence_file;

    /** @brief Output stream for the CG-iteration count CSV. */
    std::ofstream iterations_log_file;

    /** @brief Output stream for the point-probe (mid-domain) CSV. */
    std::ofstream point_probe_log_file;

    // ---- Mesh parameters --------------------------------------------------

    /** @brief Number of elements in the x- and y-directions. */
    std::pair<unsigned int, unsigned int> N_el;

    /** @brief Bounding box corners (bottom-left, top-right). */
    const std::pair<Point<dim>, Point<dim>> geometry;

    /** @brief Polynomial degree of the finite-element space. */
    const unsigned int r;

    // ---- Time parameters --------------------------------------------------

    /** @brief Final simulation time. */
    const double T;

    /** @brief Time-step size @f$ \Delta t @f$. */
    const double delta_t;

    /** @brief Current simulation time. */
    double time = 0.0;

    /** @brief Current time-step index. */
    unsigned int timestep_number = 0;

    // ---- Monitoring variables ---------------------------------------------

    /** @brief Discrete energy at the current time step. */
    double current_energy;

    /** @brief Counter for accumulated error samples (unused legacy). */
    unsigned int error_sample_count = 0;

    /** @brief Wall-clock simulation time in seconds. */
    double simulation_time = 0.0;

    // ---- Problem data (functions) -----------------------------------------

    /** @brief Wave-speed function @f$ c(\mathbf x) @f$. */
    const Function<dim>& c;

    /** @brief Forcing term @f$ f(\mathbf x,t) @f$. */
    Function<dim>& f;

    /** @brief Initial displacement @f$ u_0(\mathbf x) @f$. */
    const Function<dim>& u0;

    /** @brief Initial velocity @f$ v_0(\mathbf x) @f$. */
    const Function<dim>& v0;

    /** @brief Dirichlet boundary data @f$ g(\mathbf x,t) @f$ for @f$ u @f$. */
    Function<dim>& g;

    /** @brief Time-derivative of the Dirichlet data @f$ \partial_t g @f$. */
    Function<dim>& dgdt;

    // ---- Logging parameters -----------------------------------------------

    /** @brief Write energy / error CSVs every this many steps (0 = disabled). */
    const unsigned int log_every;

    /** @brief Print step info to stdout every this many steps. */
    const unsigned int print_every;

    // ---- Exact solution ---------------------------------------------------

    /** @brief Pointer to a manufactured exact solution (nullptr if unavailable). */
    Function<dim>* exact_solution;

    // ---- MPI --------------------------------------------------------------

    /** @brief Total number of MPI processes. */
    const unsigned int mpi_size;

    /** @brief Rank of this MPI process. */
    const unsigned int mpi_rank;

    // ---- Mesh & FE objects ------------------------------------------------

    /** @brief Fully-distributed MPI triangulation. */
    parallel::fullydistributed::Triangulation<dim> mesh;

    /** @brief Finite-element object (FE_SimplexP of degree @a r). */
    std::unique_ptr<FiniteElement<dim>> fe;

    /** @brief Gauss quadrature rule on simplices. */
    std::unique_ptr<Quadrature<dim>> quadrature;

    /** @brief Degree-of-freedom handler. */
    DoFHandler<dim> dof_handler;

    // ---- Matrices (common to both methods) --------------------------------

    /** @brief Global mass matrix @f$ M @f$. */
    TrilinosWrappers::SparseMatrix mass_matrix;

    /** @brief Global stiffness matrix @f$ K @f$ (includes @f$ c^2 @f$ weighting). */
    TrilinosWrappers::SparseMatrix stiffness_matrix;

    // ---- Solution vectors -------------------------------------------------

    /** @brief Current displacement @f$ u^{n+1} @f$. */
    TrilinosWrappers::MPI::Vector solution_u;

    /** @brief Current velocity @f$ v^{n+1} @f$. */
    TrilinosWrappers::MPI::Vector solution_v;

    /** @brief Previous displacement @f$ u^n @f$. */
    TrilinosWrappers::MPI::Vector old_solution_u;

    /** @brief Previous velocity @f$ v^n @f$. */
    TrilinosWrappers::MPI::Vector old_solution_v;

    /** @brief Right-hand side vector of the current linear system. */
    TrilinosWrappers::MPI::Vector system_rhs;

    // ---- Output -----------------------------------------------------------

    /** @brief Conditional output stream (prints only on rank 0). */
    ConditionalOStream pcout;
};

/**
 * @brief Convert a double to a filesystem-safe string.
 *
 * Trailing zeros and decimal points are removed; the dot is replaced
 * by an underscore so the result can be used in file/folder names.
 *
 * @param x         The value to convert.
 * @param precision Number of decimal digits (default 6).
 * @return Sanitised string representation (e.g. 0.25 → "0_25").
 */
std::string clean_double(double x, int precision = 6);

#endif
