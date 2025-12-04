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
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Wave
{
  public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 2;

    // Constructor.
    Wave(const std::string& mesh_file_name_,
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
        : mesh_file_name(mesh_file_name_), r(r_), T(T_), gamma(gamma_), beta(beta_), delta_t(delta_t_), c(c_), f(f_), u0(u0_), v0(v0_), g(g_), dgdt(dgdt_),
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

    // System solution.
    void
    solve_a();

    // update u and v
    void
    update_u_v();

    // Output.
    void
    output() const;

    // Name of the mesh.
    const std::string mesh_file_name;

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

    // boundary value for a (= d2g/dt2)
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

// Equation Data

// Inizial condition u0
// template <int dim>
// class InitialValuesU : public Function<dim>
// {
//   public:
//     virtual double value(const Point<dim>& p,
//                          const unsigned int /*component*/ = 0) const override
//     {
//         const double x = p[0] - 0.5;
//         const double y = p[1] - 0.5;
//         const double r2 = x * x + y * y;
//         return std::exp(-50.0 * r2);
//     }
// };

// // Initial condition v0
// template <int dim>
// class InitialValuesV : public Function<dim>
// {
//   public:
//     virtual double value(const Point<dim>& /*p*/,
//                          const unsigned int /*component*/ = 0) const override
//     {
//         return 0.0;
//     }
// };

// Wave speed function
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

// // Right hand side forcing term
// template <int dim>
// class ForcingTerm : public Function<dim>
// {
//   public:
//     virtual double value(const Point<dim>& /*p*/,
//                          const unsigned int /*component*/ = 0) const override
//     {
//         return 0.0;
//     }
// };

// // Dirichlet boundary condition
// template <int dim>
// class DirichletCondition : public Function<dim>
// {
//   public:
//     virtual double value(const Point<dim>& /*p*/,
//                          const unsigned int /*component*/ = 0) const override
//     {
//         return 0.0;
//     }

//     // Default-construct with an internally-defined derivative
//     DirichletCondition()
//         : Function<dim>(), derivative()
//     {
//     }

//     // Public member so callers can use &g.derivative
//     // Note: accessible on const instances as well
//     class DerivativeImpl : public Function<dim>
//     {
//       public:
//         virtual double value(const Point<dim>& /*p*/,
//                              const unsigned int /*component*/ = 0) const override
//         {
//             return 0.0;
//         }
//     };

//     DerivativeImpl derivative;
// };

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
