#include "ParameterReader.hpp"
#include "WaveTheta.hpp"
#include <deal.II/base/conditional_ostream.h>

int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    ConditionalOStream pcout(std::cout, rank == 0);

    // get the file name from the argument
    std::string parameters_file =
        (argc > 1) ? std::string(argv[1]) : std::string("../parameters/sine-membrane.json");
    if (argc <= 1)
    {
        pcout << "Usage:./main <path-to-arguments-file> \nRemember you are inside /build" << std::endl;
        pcout << "Using default parameter file: " << parameters_file << std::endl;
    }
    else
        pcout << "Using parameter file from argument: " << parameters_file << std::endl;
    pcout << "===============================================" << std::endl;

    // ---------------------------------------
    constexpr unsigned int dim = WaveTheta::dim;

    ParameterHandler prm;
    ParameterReader param(prm);

    FunctionParser<dim> c;
    FunctionParser<dim> f;
    FunctionParser<dim> u0;
    FunctionParser<dim> v0;
    FunctionParser<dim> g;
    FunctionParser<dim> dgdt; // derivative of g over time

    std::vector<std::string> function_names { "C", "F", "U0", "V0", "G", "DGDT" };
    param.declare(function_names);

    try
    {
        param.parse(parameters_file);
        param.load_functions(function_names, { &c, &f, &u0, &v0, &g, &dgdt });

        // Minimal debug output
        pcout << "Parsed parameters:" << std::endl;
        pcout << "  Mesh File Name: " << prm.get("Mesh File Name") << std::endl;
        pcout << "  R (degree): " << prm.get_integer("R") << std::endl;
        pcout << "  T: " << prm.get_double("T") << std::endl;
        pcout << "  Theta: " << prm.get_double("Theta") << std::endl;
        pcout << "  Dt: " << prm.get_double("Dt") << std::endl;
    }
    catch (const std::invalid_argument& e)
    {
        pcout << "Error while parsing parameters/functions: " << e.what() << std::endl;
        pcout << "Hint: check JSON fields (R, T, Theta, Dt) and function strings; ensure numeric fields are valid numbers and not empty." << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        pcout << "Unexpected error while parsing parameters: " << e.what() << std::endl;
        return 1;
    }

    try
    {
        WaveTheta problem(
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
    }
    catch (const std::invalid_argument& e)
    {
        pcout << "Error while initializing or running WaveTheta: " << e.what() << std::endl;
        pcout << "Likely cause: a non-numeric or malformed value in the parameter file (stod failure)." << std::endl;
        pcout << "Please verify fields like 'R', 'T', 'Theta', 'Dt' and function definitions C/F/U0/V0/G/DGDT in " << parameters_file << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        pcout << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
