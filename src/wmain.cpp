#include "wave.hpp"


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
