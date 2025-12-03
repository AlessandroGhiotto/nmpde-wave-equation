#include "wave.hpp"

int main(int argc, char* argv[])
{
    constexpr unsigned int dim = Wave::dim;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    InitialValuesU<dim> u0;
    InitialValuesV<dim> v0;
    WaveSpeed<dim> c;
    RightHandSide<dim> f;
    BoundaryValuesU<dim> g;
    BoundaryValuesV<dim> dgdt;

    Wave problem(
        /* mesh filename */ "../mesh/mesh-square-40.msh",
        /* degree */ 1,
        /* T */ 5.0,
        /* theta */ 0.5,
        /* delta_t */ 1. / 64,
        /* wave speed*/ c,
        /* forcing term RHS*/ f,
        /* initial u */ u0,
        /* initial v */ v0,
        /* u boundary cond */ g,
        /* v buondary cond */ dgdt);

    problem.run();

    return 0;
}
