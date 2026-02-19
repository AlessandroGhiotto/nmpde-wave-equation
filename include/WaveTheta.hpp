/**
 * @file WaveTheta.hpp
 * @brief Wave-equation solver using the theta-method time discretisation.
 *
 * Implements a second-order-in-time theta-method where the unknowns are
 * displacement @f$ u @f$ and velocity @f$ v @f$.  Two symmetric positive-definite
 * linear systems (one for @f$ u^{n+1} @f$, one for @f$ v^{n+1} @f$) are solved
 * at each time step with CG + AMG.
 *
 * This file also contains simple template helper classes for equation data
 * (wave speed, initial conditions, forcing, boundary values) that can be used
 * for quick stand-alone tests without a parameter file.
 */

#ifndef WAVE_THETA_HPP
#define WAVE_THETA_HPP

#include "WaveEquationBase.hpp"
#include <deal.II/lac/solver_cg.h>

using namespace dealii;

/**
 * @class WaveTheta
 * @brief Concrete wave-equation solver with the theta-method.
 *
 * Rewrites the second-order wave equation as a first-order system
 * @f{align*}{
 *   \partial_t u &= v, \\
 *   M\,\partial_t v &= -K\,u + f,
 * @f}
 * and applies the theta-method (@f$ \theta \in [0,1] @f$) for time integration:
 * @f{align*}{
 *   (M + \theta^2 \Delta t^2\, K)\, u^{n+1}
 *       &= M\, u^n + \Delta t\, M\, v^n
 *         - \theta(1-\theta)\,\Delta t^2\, K\, u^n + \theta\,\Delta t^2\, F_{\theta}, \\
 *   M\, v^{n+1} &= M\, v^n
 *         - \Delta t\bigl[(1-\theta)\, K\, u^n + \theta\, K\, u^{n+1}\bigr]
 *         + \Delta t\, F_{\theta},
 * @f}
 * where @f$ F_{\theta} = \theta f^{n+1} + (1-\theta) f^n @f$.
 *
 * Special cases: @f$ \theta = 0 @f$ (explicit forward Euler),
 * @f$ \theta = 1/2 @f$ (Crank–Nicolson), @f$ \theta = 1 @f$ (implicit backward Euler).
 */
class WaveTheta : public WaveEquationBase
{
  public:
    /**
     * @brief Construct the theta-method solver.
     *
     * @param problem_name_  Human-readable problem name.
     * @param N_el_          Number of mesh elements (x, y).
     * @param geometry_      Domain bounding box.
     * @param r_             FE polynomial degree.
     * @param T_             Final time.
     * @param theta_         Theta parameter @f$ \theta \in [0,1] @f$.
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
    WaveTheta(const std::string& problem_name_,
              const std::pair<unsigned int, unsigned int>& N_el_,
              const std::pair<Point<dim>, Point<dim>>& geometry_,
              const unsigned int& r_,
              const double& T_,
              const double& theta_,
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
          theta(theta_)
    {
    }

    /**
     * @brief Run the full theta-method time-stepping simulation.
     *
     * Sets up mesh and FE space, assembles @f$ M @f$ and @f$ K @f$, sets
     * initial conditions, then marches in time until @f$ t = T @f$.
     */
    void run() override;

  protected:
    // ---- Theta-specific methods -------------------------------------------

    /** @brief Initialise mesh, FE, DoF handler, sparsity pattern and vectors. */
    void setup();

    /**
     * @brief Assemble @f$ M @f$ and @f$ K @f$, then form the clean (no-BC)
     *        system matrices for @f$ u @f$ and @f$ v @f$.
     *
     * - matrix_u @f$ = M + (\theta\,\Delta t)^2\, K @f$
     * - matrix_v @f$ = M @f$
     */
    void assemble_matrices();

    /**
     * @brief Assemble the right-hand side for the displacement system.
     *
     * Includes contributions from @f$ M u^n @f$, @f$ M v^n @f$, the stiffness
     * term and the theta-weighted forcing.
     */
    void assemble_rhs_u();

    /**
     * @brief Assemble the right-hand side for the velocity system.
     *
     * Uses the freshly computed @f$ u^{n+1} @f$ in the stiffness contribution.
     */
    void assemble_rhs_v();

    /**
     * @brief Solve the displacement system for @f$ u^{n+1} @f$.
     *
     * Applies Dirichlet BCs @f$ u|_{\partial\Omega} = g(t^{n+1}) @f$ and
     * solves with CG + AMG.
     */
    void solve_u();

    /**
     * @brief Solve the velocity system for @f$ v^{n+1} @f$.
     *
     * Applies Dirichlet BCs @f$ v|_{\partial\Omega} = \partial_t g(t^{n+1}) @f$
     * and solves with CG + AMG.
     */
    void solve_v();

    // ---- Theta parameter --------------------------------------------------

    /** @brief Theta parameter @f$ \theta \in [0,1] @f$ for the time-stepping scheme. */
    const double theta;

    // ---- System matrices --------------------------------------------------

    /**
     * @brief Clean system matrix for @f$ u @f$:
     *        @f$ M + (\theta\,\Delta t)^2\, K @f$ (no BCs).
     */
    TrilinosWrappers::SparseMatrix matrix_u;

    /**
     * @brief Clean system matrix for @f$ v @f$: @f$ M @f$ (no BCs).
     */
    TrilinosWrappers::SparseMatrix matrix_v;

    /**
     * @brief BC-modified system matrix for the displacement solve.
     *
     * The AMG preconditioner keeps an internal pointer to this matrix.
     */
    TrilinosWrappers::SparseMatrix system_matrix_u;

    /**
     * @brief BC-modified system matrix for the velocity solve.
     */
    TrilinosWrappers::SparseMatrix system_matrix_v;

    // ---- Preconditioners --------------------------------------------------

    /** @brief Cached AMG preconditioner for the displacement system. */
    TrilinosWrappers::PreconditionAMG preconditioner_u;

    /** @brief Cached AMG preconditioner for the velocity system. */
    TrilinosWrappers::PreconditionAMG preconditioner_v;

    /** @brief Whether preconditioner_u has been initialised. */
    bool preconditioner_u_initialized = false;

    /** @brief Whether preconditioner_v has been initialised. */
    bool preconditioner_v_initialized = false;

    /** @brief Number of CG iterations in the last solve_u() call. */
    unsigned int current_iterations_u;

    /** @brief Number of CG iterations in the last solve_v() call. */
    unsigned int current_iterations_v;
};

// =========================================================================
// Default equation data (stand-alone test helpers)
// =========================================================================

/**
 * @brief Default wave speed: @f$ c(\mathbf x) \equiv 1 @f$.
 * @tparam dim Spatial dimension.
 */
template <int dim>
class WaveSpeed : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& /*p*/,
                         const unsigned int /*component*/ = 0) const override
    {
        return 1.0;
    }
};

/**
 * @brief Default initial displacement: @f$ u_0(\mathbf x) \equiv 0 @f$.
 * @tparam dim Spatial dimension.
 */
template <int dim>
class InitialValuesU : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& /*p*/,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }
};

/**
 * @brief Default initial velocity: @f$ v_0(\mathbf x) \equiv 0 @f$.
 * @tparam dim Spatial dimension.
 */
template <int dim>
class InitialValuesV : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& /*p*/,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }
};

/**
 * @brief Default forcing term: @f$ f(\mathbf x,t) \equiv 0 @f$.
 * @tparam dim Spatial dimension.
 */
template <int dim>
class RightHandSide : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& /*p*/,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }
};

/**
 * @brief Default Dirichlet boundary data for @f$ u @f$.
 *
 * A sinusoidal pulse is applied on a portion of the left boundary
 * @f$ (x < 0.5,\; y \in (1/3, 2/3)) @f$ for @f$ t \le 0.5 @f$.
 * Elsewhere the value is zero.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class BoundaryValuesU : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        if ((this->get_time() <= 0.5) && (p[0] < 0.5) && (p[1] > 1. / 3) &&
            (p[1] < 2. / 3))
            return std::sin(this->get_time() * 4 * numbers::PI);
        else
            return 0;
    }
};

/**
 * @brief Default Dirichlet boundary data for @f$ v = \partial_t u @f$.
 *
 * Time-derivative of BoundaryValuesU: a cosine pulse on the same
 * boundary strip for @f$ t \le 0.5 @f$, zero elsewhere.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class BoundaryValuesV : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        if ((this->get_time() <= 0.5) && (p[0] < 0.5) && (p[1] > 1. / 3) &&
            (p[1] < 2. / 3))
            return (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
        else
            return 0;
    }
};

#endif
