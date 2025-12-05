#include "ParameterReader.hpp"
#include "WaveTheta.hpp"

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

    std::string problem_name = "theta-" + std::filesystem::path(parameters_file).stem().string();
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
        pcout << "  Problem name: " << problem_name << std::endl;
        pcout << "  Geometry: " << prm.get("Geometry") << std::endl;
        pcout << "  Nel: " << prm.get("Nel") << std::endl;
        pcout << "  R (degree): " << prm.get_integer("R") << std::endl;
        pcout << "  T: " << prm.get_double("T") << std::endl;
        pcout << "  Theta: " << prm.get_double("Theta") << std::endl;
        pcout << "  Dt: " << prm.get_double("Dt") << std::endl;
    }
    catch (const std::invalid_argument& e)
    {
        pcout << "Error while parsing parameters/functions: " << e.what() << std::endl;
        pcout << "Hint: check JSON fields (Geometry, Nel, R, T, Theta, Dt) and function strings; ensure numeric fields are valid numbers and not empty." << std::endl;
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
            /* problem name */ problem_name,
            /* N_el */ param.get_nel(),
            /* geometry */ param.get_geometry(),
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
