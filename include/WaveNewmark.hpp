#ifndef WAVE_NEWMARK_HPP
#define WAVE_NEWMARK_HPP

#include "WaveEquationBase.hpp"
#include <deal.II/lac/solver_cg.h>

using namespace dealii;

/**
 * Class managing the wave equation with Newmark time discretization.
 */
class WaveNewmark : public WaveEquationBase
{
  public:
    // Constructor
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

    // Run the time-dependent simulation (implements pure virtual)
    void run() override;

  protected:
    // Newmark-specific methods
    void setup();
    void assemble_matrices();
    void assemble_rhs();
    void solve_a();
    void update_u_v();
    void apply_dirichlet_bc();

    // Newmark parameters
    const double gamma;
    const double beta;

    // Additional vector for acceleration (specific to Newmark)
    TrilinosWrappers::MPI::Vector solution_a;
    TrilinosWrappers::MPI::Vector old_solution_a;

    // System matrix (specific to Newmark)
    TrilinosWrappers::SparseMatrix matrix_a;

    // Cached AMG preconditioner (built once on matrix_a, reused every time step)
    TrilinosWrappers::PreconditionAMG preconditioner_a;

    unsigned int current_iterations;
};

#endif
