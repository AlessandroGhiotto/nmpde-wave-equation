#include "ParameterReader.hpp"
#include "WaveNewmark.hpp"
#include <cstdlib>
#include <mpi.h>

int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    ConditionalOStream pcout(std::cout, rank == 0);

    if (rank == 0)
    {
        int len = 0;
        char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
        if (MPI_Get_library_version(version, &len) == MPI_SUCCESS)
            pcout << "MPI library: " << std::string(version, (len > 0 ? len : 0)) << std::endl;

        pcout << "MPI world size: " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
    }

    pcout << "===============================================" << std::endl;
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

    // Make the parameter file path available to downstream code (e.g., for logging/copying).
    ::setenv("NMPDE_PARAM_FILE", parameters_file.c_str(), 1);

    // ---------------------------------------

    constexpr unsigned int dim = WaveNewmark::dim;

    std::string problem_name = "newmark-" + std::filesystem::path(parameters_file).stem().string();
    ParameterHandler prm;
    ParameterReader param(prm);

    FunctionParser<dim> c;
    FunctionParser<dim> f;
    FunctionParser<dim> u0;
    FunctionParser<dim> v0;
    FunctionParser<dim> g;
    FunctionParser<dim> dgdt; // derivative of g over time

    FunctionParser<dim> exact_solution;

    std::vector<std::string> function_names { "C", "F", "U0", "V0", "G", "DGDT", "Solution" };
    param.declare(function_names);

    try
    {
        param.parse(parameters_file);
        param.load_functions(function_names, { &c, &f, &u0, &v0, &g, &dgdt, &exact_solution });

        // Minimal debug output
        pcout << "Parsed parameters:" << std::endl;
        pcout << "  Problem name: " << problem_name << std::endl;
        pcout << "  Geometry: " << prm.get("Geometry") << std::endl;
        pcout << "  Nel: " << prm.get("Nel") << std::endl;
        pcout << "  R (degree): " << prm.get_integer("R") << std::endl;
        pcout << "  T: " << prm.get_double("T") << std::endl;
        pcout << "  Beta: " << prm.get_double("Beta") << std::endl;
        pcout << "  Gamma: " << prm.get_double("Gamma") << std::endl;
        pcout << "  Dt: " << prm.get_double("Dt") << std::endl;
    }
    catch (const std::invalid_argument& e)
    {
        pcout << "Error while parsing parameters/functions: " << e.what() << std::endl;
        pcout << "Hint: check JSON fields (Geometry, Nel, R, T, Beta, Gamma, Dt) and function strings; ensure numeric fields are valid numbers and not empty." << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        pcout << "Unexpected error while parsing parameters: " << e.what() << std::endl;
        return 1;
    }

    // Export runtime flags (used by WaveEquationBase without changing class APIs).
    const bool save_solution = prm.get_bool("Save Solution");
    const bool enable_logging = prm.get_bool("Enable Logging");
    ::setenv("NMPDE_SAVE_SOLUTION", save_solution ? "1" : "0", 1);
    // NOTE: logging is controlled solely via log_every (0 disables), no env var.

    int log_every = prm.get_integer("Log Every");
    if (!enable_logging)
        log_every = 0;

    ::setenv("NMPDE_LOG_EVERY", std::to_string(log_every).c_str(), 1);

    // Check if exact_solution was initialized, if not set it to nullptr
    Function<dim>* exact_solution_ptr = nullptr;
    try
    {
        prm.enter_subsection("Solution");
        if (!prm.get("Function expression").empty())
        {
            exact_solution_ptr = &exact_solution;
        }
        prm.leave_subsection();
    }
    catch (...)
    {
        // If subsection doesn't exist, exact_solution_ptr remains nullptr
    }

    try
    {
        WaveNewmark problem(
            /* problem name */ problem_name,
            /* N_el */ param.get_nel(),
            /* geometry */ param.get_geometry(),
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
            /* v buondary cond */ dgdt,
            /* log every */ log_every,
            /* print every */ prm.get_integer("Print Every"),
            /* exact solution */ exact_solution_ptr);

        problem.run();
    }
    catch (const std::invalid_argument& e)
    {
        pcout << "Error while initializing or running WaveNewmark: " << e.what() << std::endl;
        pcout << "Likely cause: a non-numeric or malformed value in the parameter file (stod failure)." << std::endl;
        pcout << "Please verify fields like 'R', 'T', 'Beta', 'Gamma', 'Dt' and function definitions C/F/U0/V0/G/DGDT in " << parameters_file << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        pcout << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
