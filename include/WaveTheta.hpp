#ifndef WAVE_THETA_HPP
#define WAVE_THETA_HPP

#include "WaveEquationBase.hpp"
#include <deal.II/lac/solver_cg.h>

using namespace dealii;

/**
 * Class managing the wave equation with theta-method time discretization.
 */
class WaveTheta : public WaveEquationBase
{
  public:
    // Constructor
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

    // Run the time-dependent simulation (implements pure virtual)
    void run() override;

  protected:
    // Theta-specific methods
    void setup();
    void assemble_matrices();
    void assemble_rhs_u();
    void assemble_rhs_v();
    void solve_u();
    void solve_v();

    // Theta parameter for the theta method
    const double theta;

    // System matrices (specific to theta-method)
    TrilinosWrappers::SparseMatrix matrix_u;
    TrilinosWrappers::SparseMatrix matrix_v;

    unsigned int current_iterations_u;
    unsigned int current_iterations_v;
};

// Equation Data (keep existing template classes)
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
