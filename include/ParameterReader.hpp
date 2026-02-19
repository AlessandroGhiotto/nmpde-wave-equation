/**
 * @file ParameterReader.hpp
 * @brief Wrapper around deal.II's ParameterHandler for parsing JSON/PRM
 *        parameter files used by the wave-equation solvers.
 *
 * Declares scalar parameters (mesh size, polynomial degree, time-stepping
 * constants, …) and function subsections (wave speed, forcing, initial /
 * boundary data) and loads them into FunctionParser objects.
 */

#ifndef PARAMETER_READER_HPP
#define PARAMETER_READER_HPP

#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace dealii;

/** @brief Spatial dimension assumed by the parameter reader. */
static constexpr unsigned int dim = 2;

/**
 * @class ParameterReader
 * @brief Helper that declares, parses and loads simulation parameters.
 *
 * Usage pattern:
 * @code
 *   ParameterHandler prm;
 *   ParameterReader reader(prm);
 *   reader.declare({"C", "F", "U0", "V0", "G", "DGDT", "Solution"});
 *   reader.parse("parameters.json");
 *   reader.load_functions(names, func_ptrs);
 * @endcode
 */
class ParameterReader
{
  public:
    /**
     * @brief Construct a reader bound to an existing ParameterHandler.
     * @param paramhandler  Reference to the deal.II ParameterHandler instance.
     */
    ParameterReader(ParameterHandler& paramhandler);

    /**
     * @brief Declare all scalar entries and function subsections.
     * @param function_names  Names of the function subsections to declare
     *                        (e.g. "C", "F", "U0", …).
     */
    void declare(const std::vector<std::string>& function_names);

    /**
     * @brief Parse a parameter file (JSON or PRM format).
     * @param filename  Path to the parameter file.
     */
    void parse(const std::string& filename);

    /**
     * @brief Initialise FunctionParser objects from the parsed subsections.
     *
     * Each name in @p names must match a previously declared subsection.
     * The corresponding FunctionParser pointer in @p funcs is initialised
     * with the expression and constants read from the file.  The special
     * subsection "Solution" is silently skipped when its expression is empty.
     *
     * @param names  Subsection names (same order as @p funcs).
     * @param funcs  Pointers to FunctionParser objects to initialise.
     */
    void load_functions(const std::vector<std::string>& names,
                        const std::vector<FunctionParser<dim>*>& funcs);

    /**
     * @brief Read the "Nel" entry and return (N_el_x, N_el_y).
     *
     * A single value is duplicated for a square grid; two comma-separated
     * values give a rectangular grid.
     *
     * @return Pair of element counts (x, y).
     */
    std::pair<unsigned int, unsigned int> get_nel() const;

    /**
     * @brief Read the "Geometry" entry and return the bounding box.
     *
     * Expected format: @c "[x_min, x_max] x [y_min, y_max]".
     *
     * @return Pair of Point<dim> (bottom-left, top-right).
     */
    std::pair<Point<dim>, Point<dim>> get_geometry() const;

  private:
    /** @brief Declare all scalar entries (Nel, R, T, Dt, Theta, …). */
    void declare_scalar_parameters();

    /**
     * @brief Declare function subsections for the given names.
     * @param names  Subsection names.
     */
    void declare_function_subsections(const std::vector<std::string>& names);

    /** @brief Reference to the underlying deal.II ParameterHandler. */
    ParameterHandler& prm;
};

#endif // PARAMETER_READER_HPP
