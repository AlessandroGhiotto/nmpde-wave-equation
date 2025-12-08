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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace dealii;

class WaveEquationBase
{
  public:
    static constexpr unsigned int dim = 2;

    virtual ~WaveEquationBase() = default;

    // Pure virtual method for running simulation (implemented by derived classes)
    virtual void run() = 0;

  protected:
    // Constructor
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

    // Common methods
    void setup_mesh();
    void setup_fe();
    void setup_dof_handler();

    void prepare_output_filename(const std::string& method_params);
    void compute_and_log_energy();
    void compute_and_log_error();
    void compute_final_errors();
    void print_step_info();
    void output() const;

    double compute_error(const VectorTools::NormType&, const Function<dim>&) const;
    double compute_relative_error(const VectorTools::NormType&, const Function<dim>&) const;

    // Problem description
    const std::string problem_name;
    std::string output_folder;
    std::ofstream energy_log_file;
    std::ofstream error_log_file;
    std::ofstream convergence_file;

    // Mesh parameters
    std::pair<unsigned int, unsigned int> N_el;
    const std::pair<Point<dim>, Point<dim>> geometry;
    const unsigned int r;

    // Time parameters
    const double T;
    const double delta_t;
    double time = 0.0;
    unsigned int timestep_number = 0;

    // Monitoring variables
    double current_energy;
    double accumulated_L2_error = 0.0;
    double accumulated_H1_error = 0.0;
    unsigned int error_sample_count = 0;

    // Problem data
    const Function<dim>& c;
    Function<dim>& f;
    const Function<dim>& u0;
    const Function<dim>& v0;
    Function<dim>& g;
    Function<dim>& dgdt;

    // Logging parameters
    const unsigned int log_every;
    const unsigned int print_every;

    // Solution
    Function<dim>* exact_solution;

    // MPI
    const unsigned int mpi_size;
    const unsigned int mpi_rank;

    // Mesh and FE
    parallel::fullydistributed::Triangulation<dim> mesh;
    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<Quadrature<dim>> quadrature;
    DoFHandler<dim> dof_handler;

    // Matrices (common to both methods)
    TrilinosWrappers::SparseMatrix mass_matrix;
    TrilinosWrappers::SparseMatrix stiffness_matrix;

    // Solution vectors
    TrilinosWrappers::MPI::Vector solution_u, solution_v;
    TrilinosWrappers::MPI::Vector old_solution_u, old_solution_v;
    TrilinosWrappers::MPI::Vector system_rhs;

    // Output
    ConditionalOStream pcout;
};

std::string clean_double(double, int precision = 6);

#endif
