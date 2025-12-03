#ifndef PARAMETER_READER_HPP
#define PARAMETER_READER_HPP

#include "wave.hpp" // for Wave::dim

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

// Uses Wave::dim to bind parsed functions to the current problem dimension.
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
                        const std::vector<Functions::ParsedFunction<Wave::dim>*>& funcs);

    // Helper to extract Nel list
    std::vector<unsigned int> get_Nel_list() const;

  private:
    void declare_scalar_parameters();
    void declare_function_subsections(const std::vector<std::string>& names);

    ParameterHandler& prm;
};

#endif // PARAMETER_READER_HPP
