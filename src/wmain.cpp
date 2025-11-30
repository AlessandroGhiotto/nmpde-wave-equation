#include "wave.hpp"

// Classe per la condizione iniziale u0
template <int dim>
class InitialU : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& p,
                         const unsigned int /*component*/ = 0) const override
    {
        const double x = p[0] - 0.5;
        const double y = p[1] - 0.5;
        const double r2 = x * x + y * y;
        return std::exp(-50.0 * r2);
    }
};

// Classe per la condizione iniziale v0
template <int dim>
class InitialV : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& /*p*/,
                         const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
};

// Classe per la velocità dell'onda c
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

// Classe per il forcing term f(x,t)
template <int dim>
class ForcingTerm : public Function<dim>
{
  public:
    virtual double value(const Point<dim>& /*p*/,
                         const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
};

int main(int argc, char* argv[])
{
    constexpr unsigned int dim = Wave::dim;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    InitialU<dim> u0;
    InitialV<dim> v0;
    WaveSpeed<dim> c;
    ForcingTerm<dim> f;

    Wave problem(
        /* mesh filename */ "../mesh/mesh-square-40.msh",
        /* degree */ 1,
        /* T */ 1.0,
        /* theta */ 0.5,
        /* delta_t */ 0.0025,
        c,
        f,
        u0,
        v0);

    problem.run();

    return 0;
}
