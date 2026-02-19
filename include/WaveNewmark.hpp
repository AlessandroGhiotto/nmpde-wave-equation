/**
 * @file WaveNewmark.hpp
 * @brief Wave-equation solver using the Newmark-beta time-integration scheme.
 *
 * Implements the classical Newmark family of methods parameterised by
 * @f$ \gamma @f$ and @f$ \beta @f$.  The unknowns are displacement @f$ u @f$,
 * velocity @f$ v @f$ and acceleration @f$ a @f$.  At each time step one
 * symmetric positive-definite system for @f$ a^{n+1} @f$ is solved with CG +
 * AMG, then @f$ u @f$ and @f$ v @f$ are updated algebraically.
 */

#ifndef WAVE_NEWMARK_HPP
#define WAVE_NEWMARK_HPP

#include "WaveEquationBase.hpp"
#include <deal.II/lac/solver_cg.h>

using namespace dealii;

/**
 * @class WaveNewmark
 * @brief Concrete wave-equation solver with the Newmark-@f$\beta@f$ method.
 *
 * The Newmark update reads:
 * @f{align*}{
 *   u^{n+1} &= u^n + \Delta t\, v^n
 *             + \Delta t^2 \bigl[(\tfrac12 - \beta)\, a^n + \beta\, a^{n+1}\bigr], \\
 *   v^{n+1} &= v^n + \Delta t \bigl[(1 - \gamma)\, a^n + \gamma\, a^{n+1}\bigr],
 * @f}
 * where @f$ a^{n+1} @f$ is obtained from the linear system
 * @f$ (M + \beta\,\Delta t^2\, K)\, a^{n+1} = f^{n+1} - K\, z @f$
 * with @f$ z = u^n + \Delta t\, v^n + \Delta t^2 (\tfrac12 - \beta)\, a^n @f$.
 *
 * Common parameter choices:
 * - @f$ \gamma = 1/2,\; \beta = 1/4 @f$  (average-acceleration, unconditionally stable)
 * - @f$ \gamma = 1/2,\; \beta = 0 @f$    (explicit central differences)
 */
class WaveNewmark : public WaveEquationBase
{
  public:
    /**
     * @brief Construct the Newmark solver.
     *
     * @param problem_name_  Human-readable problem name.
     * @param N_el_          Number of mesh elements (x, y).
     * @param geometry_      Domain bounding box.
     * @param r_             FE polynomial degree.
     * @param T_             Final time.
     * @param gamma_         Newmark parameter @f$ \gamma @f$.
     * @param beta_          Newmark parameter @f$ \beta @f$.
     * @param delta_t_       Time-step size.
     * @param c_             Wave-speed function.
     * @param f_             Forcing term.
     * @param u0_            Initial displacement.
     * @param v0_            Initial velocity.
     * @param g_             Dirichlet BC for @f$ u @f$.
     * @param dgdt_          Time-derivative of the Dirichlet data.
     * @param log_every_     Logging frequency (0 = off).
     * @param print_every_   Console output frequency.
     * @param exact_solution_ Pointer to exact solution (nullptr if unavailable).
     */
    WaveNewmark(
        const std::string& problem_name_,
        const std::pair<unsigned int, unsigned int>& N_el_,
        const std::pair<Point<dim>, Point<dim>>& geometry_,
        const unsigned int& r_,
        const double& T_,
        const double& gamma_,
        const double& beta_,
        const double& delta_t_,
        const Function<dim>& c_,
        Function<dim>& f_,
        const Function<dim>& u0_,
        const Function<dim>& v0_,
        Function<dim>& g_,
        Function<dim>& dgdt_,
        const unsigned int log_every_ = 10,
        const unsigned int print_every_ = 10,
        Function<dim>* exact_solution_ = nullptr)
        : WaveEquationBase(problem_name_, N_el_, geometry_, r_, T_, delta_t_,
                           c_, f_, u0_, v0_, g_, dgdt_, log_every_, print_every_, exact_solution_),
          gamma(gamma_), beta(beta_)
    {
    }

    /**
     * @brief Run the full Newmark time-stepping simulation.
     *
     * Sets up mesh and FE space, assembles @f$ M @f$ and @f$ K @f$, computes
     * a consistent initial acceleration @f$ a^0 @f$, then marches in time
     * until @f$ t = T @f$.
     */
    void run() override;

  protected:
    // ---- Newmark-specific methods -----------------------------------------

    /** @brief Initialise mesh, FE, DoF handler, sparsity pattern and vectors. */
    void setup();

    /**
     * @brief Assemble the global mass and stiffness matrices and form the
     *        clean (no-BC) system matrix @f$ M + \beta\,\Delta t^2\, K @f$.
     */
    void assemble_matrices();

    /**
     * @brief Assemble the right-hand side for the acceleration system.
     *
     * Computes @f$ \mathrm{rhs} = f^{n+1} - K\,z @f$ where
     * @f$ z = u^n + \Delta t\, v^n + \Delta t^2 (\tfrac12-\beta)\, a^n @f$.
     */
    void assemble_rhs();

    /**
     * @brief Solve the linear system for @f$ a^{n+1} @f$ and apply Dirichlet BCs.
     *
     * Uses CG with an AMG preconditioner.  The preconditioner is built once
     * on the first call and reused for subsequent time steps.
     */
    void solve_a();

    /**
     * @brief Update displacement and velocity from the newly computed acceleration.
     *
     * Applies the algebraic Newmark update formulas for @f$ u^{n+1} @f$ and
     * @f$ v^{n+1} @f$.
     */
    void update_u_v();

    /**
     * @brief Apply Dirichlet boundary conditions.
     * @note  BCs are applied inside solve_a(); this is a legacy stub.
     */
    void apply_dirichlet_bc();

    // ---- Newmark parameters -----------------------------------------------

    /** @brief Newmark parameter @f$ \gamma @f$ (velocity weighting). */
    const double gamma;

    /** @brief Newmark parameter @f$ \beta @f$ (displacement weighting). */
    const double beta;

    // ---- Additional vectors for acceleration ------------------------------

    /** @brief Current acceleration @f$ a^{n+1} @f$. */
    TrilinosWrappers::MPI::Vector solution_a;

    /** @brief Previous acceleration @f$ a^n @f$. */
    TrilinosWrappers::MPI::Vector old_solution_a;

    // ---- System matrices --------------------------------------------------

    /**
     * @brief Clean system matrix @f$ M + \beta\,\Delta t^2\, K @f$ (no BCs).
     *
     * Copied into @ref system_matrix_a before applying boundary values.
     */
    TrilinosWrappers::SparseMatrix matrix_a;

    /**
     * @brief BC-modified system matrix.
     *
     * The AMG preconditioner keeps an internal pointer to this matrix, so it
     * must not be reinitialised after the preconditioner is built.
     */
    TrilinosWrappers::SparseMatrix system_matrix_a;

    // ---- Preconditioner ---------------------------------------------------

    /** @brief Cached AMG preconditioner for the acceleration system. */
    TrilinosWrappers::PreconditionAMG preconditioner_a;

    /** @brief Whether preconditioner_a has been initialised. */
    bool preconditioner_a_initialized = false;

    /** @brief Number of CG iterations in the last solve_a() call. */
    unsigned int current_iterations;
};

#endif
