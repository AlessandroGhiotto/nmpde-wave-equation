#include "ParameterReader.hpp"
#include "wave.hpp"

int main(int argc, char* argv[])
{

    // get the file name from the argument
    std::string parameters_file =
        (argc > 1) ? std::string(argv[1]) : std::string("../parameters/sine-membrane.json");
    if (argc <= 1)
    {
        std::cout << "Usage:./main <path-to-arguments-file> \nRemember you are inside /build" << std::endl;
        std::cout << "Using default parameter file: " << parameters_file << std::endl;
    }
    else
        std::cout << "Using parameter file from argument: " << parameters_file << std::endl;
    std::cout << "===============================================" << std::endl;

    // ---------------------------------------

    constexpr unsigned int dim = Wave::dim;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    ParameterHandler prm;
    ParameterReader param(prm);

    Functions::ParsedFunction<dim> u0;
    Functions::ParsedFunction<dim> v0;
    Functions::ParsedFunction<dim> c;
    Functions::ParsedFunction<dim> f;
    Functions::ParsedFunction<dim> g;
    Functions::ParsedFunction<dim> dgdt;

    std::vector<std::string> function_names { "C", "F", "U0", "V0", "G", "DGDT" };
    param.declare(function_names);
    param.parse(parameters_file);
    param.load_functions(function_names, { &c, &f, &u0, &v0, &g, &dgdt });

    Wave problem(
        /* mesh filename */ prm.get("Mesh File Name"),
        /* degree */ prm.get_integer("R"),
        /* T */ prm.get_double("T"),
        /* theta */ prm.get_double("Theta"),
        /* delta_t */ prm.get_double("Dt"),
        /* wave speed*/ c,
        /* forcing term RHS*/ f,
        /* initial u */ u0,
        /* initial v */ v0,
        /* u boundary cond */ g,
        /* v buondary cond */ dgdt);

    problem.run();

    return 0;
}
