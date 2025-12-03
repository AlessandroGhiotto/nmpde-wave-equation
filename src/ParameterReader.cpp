#include "ParameterReader.hpp"

#include "wave.hpp"

static constexpr unsigned int dim = Wave::dim;

// class ParameterReader
// {
//   public:
//     ParameterReader(ParameterHandler& paramhandler) : prm(paramhandler)
//     {
//     }

//     // Declare scalar + function subsections
//     void declare(const std::vector<std::string>& function_names);

//     // Parse file after declaration
//     void parse(const std::string& filename);

//     // Load (parse) function definitions into objects
//     void load_functions(const std::vector<std::string>& names,
//                         const std::vector<Functions::ParsedFunction<dim>*>& funcs);

//     // Helper to extract Nel list
//     std::vector<unsigned int> get_Nel_list() const;

//   private:
//     void declare_scalar_parameters();
//     void declare_function_subsections(const std::vector<std::string>& names);

//     ParameterHandler& prm;
// };

ParameterReader::ParameterReader(ParameterHandler& paramhandler)
    : prm(paramhandler)
{
}

void ParameterReader::declare_scalar_parameters()
{
    // prm.declare_entry("Nel",
    //                   "10, 20, 40, 80, 160, 320",
    //                   Patterns::List(Patterns::Integer(1), 1),
    //                   "Number of elements in a face of the 2D mesh");

    prm.declare_entry("Mesh File Name",
                      "../mesh/mesh-square-40.msh",
                      Patterns::FileName(),
                      "Path to the mesh file");
    prm.declare_entry("R",
                      "1",
                      Patterns::Integer(1),
                      "Degree of the polynomial");
    prm.declare_entry("T",
                      "1.0",
                      Patterns::Double(0.0),
                      "Length of the time interval");
    prm.declare_entry("Theta",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Theta parameter for the theta-method for time integration");
    prm.declare_entry("Dt",
                      "0.01",
                      Patterns::Double(0.0),
                      "Length of the time step");
}

void ParameterReader::declare_function_subsections(const std::vector<std::string>& names)
{
    for (const auto& n : names)
    {
        prm.enter_subsection(n);
        Functions::ParsedFunction<dim>::declare_parameters(prm, dim);
        prm.leave_subsection();
    }
}

void ParameterReader::declare(const std::vector<std::string>& function_names)
{
    declare_scalar_parameters();
    declare_function_subsections(function_names);
}

void ParameterReader::parse(const std::string& filename)
{
    prm.parse_input(filename);
}

void ParameterReader::load_functions(const std::vector<std::string>& names,
                                     const std::vector<Functions::ParsedFunction<dim>*>& funcs)
{
    if (names.size() != funcs.size())
    {
        std::cerr << "Mismatch names/functions size\n";
        return;
    }
    for (unsigned int i = 0; i < names.size(); ++i)
    {
        prm.enter_subsection(names[i]);
        funcs[i]->parse_parameters(prm);
        prm.leave_subsection();
    }
}

std::vector<unsigned int> ParameterReader::get_Nel_list() const
{
    const auto nel_str = prm.get("Nel");
    const auto tokens = Utilities::split_string_list(nel_str); // split
    std::vector<unsigned int> values;
    values.reserve(tokens.size());
    for (const auto& t : tokens)
        values.push_back(static_cast<unsigned int>(std::stoul(t))); // Convert string to unsigned integer
    return values;
}
