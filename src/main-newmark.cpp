#include "ParameterReader.hpp"
#include "WaveNewmark.hpp"

int main(int argc, char* argv[])
{

    // get the file name from the argument
    std::string parameters_file =
        (argc > 1) ? std::string(argv[1]) : std::string("../parameters/sine-membrane-newmark.json");
    if (argc <= 1)
    {
        std::cout << "Usage:./main <path-to-arguments-file> \nRemember you are inside /build" << std::endl;
        std::cout << "Using default parameter file: " << parameters_file << std::endl;
    }
    else
        std::cout << "Using parameter file from argument: " << parameters_file << std::endl;
    std::cout << "===============================================" << std::endl;

    // ---------------------------------------

    constexpr unsigned int dim = WaveNewmark::dim;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    ParameterHandler prm;
    ParameterReader param(prm);

    FunctionParser<dim> c;
    FunctionParser<dim> f;
    FunctionParser<dim> u0;
    FunctionParser<dim> v0;
    FunctionParser<dim> g;
    FunctionParser<dim> dgdt; // derivative of g over time

    std::vector<std::string> function_names {
        "C", "F", "U0", "V0", "G", "DGDT"
    };
    param.declare(function_names);

    try
    {
        param.parse(parameters_file);
        param.load_functions(function_names, { &c, &f, &u0, &v0, &g, &dgdt });

        // Minimal debug output to catch bad numeric fields from JSON
        std::cout << "Parsed parameters:" << std::endl;
        std::cout << "  Mesh File Name: " << prm.get("Mesh File Name") << std::endl;
        std::cout << "  R (degree): " << prm.get_integer("R") << std::endl;
        std::cout << "  T: " << prm.get_double("T") << std::endl;
        std::cout << "  Beta: " << prm.get_double("Beta") << std::endl;
        std::cout << "  Gamma: " << prm.get_double("Gamma") << std::endl;
        std::cout << "  Dt: " << prm.get_double("Dt") << std::endl;
    }
    catch (const std::invalid_argument& e)
    {
        std::cerr << "Error while parsing parameters/functions: " << e.what() << std::endl;
        std::cerr << "Hint: check JSON fields (R, T, Beta, Gamma, Dt) and function strings; ensure numeric fields are valid numbers and not empty." << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Unexpected error while parsing parameters: " << e.what() << std::endl;
        return 1;
    }

    try
    {
        WaveNewmark problem(
            /* mesh filename */ prm.get("Mesh File Name"),
            /* degree */ prm.get_integer("R"),
            /* T */ prm.get_double("T"),
            /* gamma */ prm.get_double("Gamma"),
            /* beta */ prm.get_double("Beta"),
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
        std::cerr << "Error while initializing or running WaveNewmark: " << e.what() << std::endl;
        std::cerr << "Likely cause: a non-numeric or malformed value in the parameter file (stod failure)." << std::endl;
        std::cerr << "Please verify fields like 'R', 'T', 'Theta', 'Dt' and function definitions C/F/U0/V0/G/DGDT in " << parameters_file << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
