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

static constexpr unsigned int dim = 2;

class ParameterReader
{
  public:
    ParameterReader(ParameterHandler& paramhandler);

    // Declare scalar + function subsections
    void declare(const std::vector<std::string>& function_names);

    // Parse file after declaration
    void parse(const std::string& filename);

    // Load (parse) function definitions into objects
    void load_functions(const std::vector<std::string>& names,
                        const std::vector<FunctionParser<dim>*>& funcs);

    // Helper to extract N_el_x and N_el_y
    std::pair<unsigned int, unsigned int> get_nel() const;

    // Helper to extract Geometry
    std::pair<Point<dim>, Point<dim>> get_geometry() const;

  private:
    void declare_scalar_parameters();
    void declare_function_subsections(const std::vector<std::string>& names);

    ParameterHandler& prm;
};

#endif // PARAMETER_READER_HPP
