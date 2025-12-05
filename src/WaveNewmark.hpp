#ifndef WAVE_NEWMARK_HPP
#define WAVE_NEWMARK_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class WaveNewmark
{
  public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 2;

    // Constructor.
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
        Function<dim>& dgdt_)
        : problem_name(problem_name_), N_el(N_el_), geometry(geometry_), r(r_), T(T_), gamma(gamma_), beta(beta_), delta_t(delta_t_), c(c_), f(f_), u0(u0_), v0(v0_), g(g_), dgdt(dgdt_),
          mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0)
    {
    }

    // Run the time-dependent simulation.
    void
    run();

  protected:
    // Initialization.
    void
    setup();

    // Assembly of matrices (mass and laplace)
    void
    assemble_matrices();

    // Assembly rhs
    void
    assemble_rhs();

    // Apply Dirichlet BCs
    void
    apply_dirichlet_bc();

    // System solution.
    void
    solve_a();

    // update u and v
    void
    update_u_v();

    // compute output filename
    void prepare_output_filename();

    // Output.
    void output() const;

    // Name of the problem and output folder
    const std::string problem_name;
    std::string output_folder;

    // Number of elements in x and y directions.
    std::pair<unsigned int, unsigned int> N_el;

    // Geometry of the domain
    const std::pair<Point<dim>, Point<dim>> geometry;

    // Polynomial degree.
    const unsigned int r;

    // Final time.
    const double T;

    // Parameters for the newmark method.
    const double gamma;
    const double beta;

    // Time step.
    const double delta_t;

    // Current time.
    double time = 0.0;

    // Current timestep number.
    unsigned int timestep_number = 0;

    // wave speed
    const Function<dim>& c;

    // Forcing term f(x,t)
    Function<dim>& f;

    // Initial conditions
    const Function<dim>& u0;

    const Function<dim>& v0;

    // boundary value for u
    Function<dim>& g;

    // boundary value for v (= dg/dt)
    Function<dim>& dgdt;

    // Number of MPI processes.
    const unsigned int mpi_size;

    // Rank of the current MPI process.
    const unsigned int mpi_rank;

    // Triangulation.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // Mass and stiffness matrices
    TrilinosWrappers::SparseMatrix mass_matrix;
    TrilinosWrappers::SparseMatrix stiffness_matrix;
    TrilinosWrappers::SparseMatrix matrix_a;

    // System right-hand side.
    TrilinosWrappers::MPI::Vector system_rhs;

    // Solution vectors with ghost elements
    TrilinosWrappers::MPI::Vector solution_u, solution_v, solution_a;
    TrilinosWrappers::MPI::Vector old_solution_u, old_solution_v, old_solution_a;

    // Output stream for process 0.
    ConditionalOStream pcout;
};

std::string clean_double(double, int precision = 6);

#endif
