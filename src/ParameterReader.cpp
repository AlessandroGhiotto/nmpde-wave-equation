#include "ParameterReader.hpp"

double parse_value_with_pi(std::string value);
std::map<std::string, double>
parse_constants_with_pi_and_multiplication(const std::string& s);

ParameterReader::ParameterReader(ParameterHandler& paramhandler)
    : prm(paramhandler)
{
}

void ParameterReader::declare_scalar_parameters()
{
    prm.declare_entry("Nel",
                      "40",
                      Patterns::List(Patterns::Integer(1), 1),
                      "Number of elements in the 2D mesh. Use single value for square grid (e.g., '40') or two space-separated values for rectangular grid (e.g., '40 50')");

    prm.declare_entry("Geometry",
                      "[0.0, 1.0] x [0.0, 1.0]",
                      Patterns::Anything(),
                      "Geometry of the domain, in the format [x_min, x_max] x [y_min, y_max]");

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
                      "Theta parameter for the theta-method for time discretization");
    prm.declare_entry("Beta",
                      "0.25",
                      Patterns::Double(0.0, 1.0),
                      "Beta parameter for the newmark method for time discretization");
    prm.declare_entry("Gamma",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Gamma parameter for the newmark method for time discretization");
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
        prm.declare_entry("Function constants",
                          "",
                          Patterns::Anything(),
                          "List of function constants");
        prm.declare_entry("Function expression",
                          "0.0",
                          Patterns::Anything(),
                          "Function expression");
        prm.declare_entry("Variable names",
                          "",
                          Patterns::Anything(),
                          "List of variable names");
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
                                     const std::vector<FunctionParser<dim>*>& funcs)
{
    if (names.size() != funcs.size())
    {
        std::cerr << "Mismatch names/functions size\n";
        return;
    }
    for (unsigned int i = 0; i < names.size(); ++i)
    {
        prm.enter_subsection(names[i]);
        auto constants = parse_constants_with_pi_and_multiplication(prm.get("Function constants"));
        constants["pi"] = numbers::PI;
        bool time_dependent = (prm.get("Variable names").find("t") != std::string::npos);
        funcs[i]->initialize(prm.get("Variable names"),
                             prm.get("Function expression"),
                             constants,
                             time_dependent);
        prm.leave_subsection();
    }
}

std::pair<Point<dim>, Point<dim>> ParameterReader::get_geometry() const
{
    const auto geom_str = prm.get("Geometry");
    // Expected format: [x_min, x_max] x [y_min, y_max]
    std::regex pattern(R"(\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]\s*x\s*\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\])");
    std::smatch match;
    if (std::regex_match(geom_str, match, pattern))
    {
        double x_min = std::stod(match[1].str());
        double x_max = std::stod(match[2].str());
        double y_min = std::stod(match[3].str());
        double y_max = std::stod(match[4].str());

        return { Point<dim>(x_min, y_min), Point<dim>(x_max, y_max) };
    }
    else
    {
        throw std::invalid_argument("Invalid Geometry format in parameters.");
    }
}

std::pair<unsigned int, unsigned int> ParameterReader::get_nel() const
{
    const auto nel_str = prm.get("Nel");

    // Trim whitespace
    auto trim = [](std::string& x) {
        x.erase(0, x.find_first_not_of(" \t"));
        x.erase(x.find_last_not_of(" \t") + 1);
    };
    std::string trimmed = nel_str;
    trim(trimmed);

    // Split by comma
    std::vector<std::string> tokens = Utilities::split_string_list(trimmed, ",");

    if (tokens.size() == 1)
    {
        // Single value: duplicate for square grid
        unsigned int nel = static_cast<unsigned int>(std::stoul(tokens[0]));
        return { nel, nel };
    }
    else if (tokens.size() == 2)
    {
        // Two values: x and y
        unsigned int nel_x = static_cast<unsigned int>(std::stoul(tokens[0]));
        unsigned int nel_y = static_cast<unsigned int>(std::stoul(tokens[1]));
        return { nel_x, nel_y };
    }
    else
    {
        throw std::invalid_argument("Invalid Nel format. Expected single value or two space-separated values.");
    }
}

// ----------------------------
// Helper functions for parsing function constants
// ----------------------------
double parse_value_with_pi(std::string value)
{
    // trim
    auto trim = [](std::string& x) {
        x.erase(0, x.find_first_not_of(" \t"));
        x.erase(x.find_last_not_of(" \t") + 1);
    };
    trim(value);

    // Case 1: pure pi
    std::string v_lower = value;
    std::transform(v_lower.begin(), v_lower.end(), v_lower.begin(), ::tolower);
    if (v_lower == "pi")
        return dealii::numbers::PI;

    // Case 2: simple form: number * pi
    std::regex mul_pattern(R"(^\s*([0-9]*\.?[0-9]+)\s*\*\s*(pi)\s*$)",
                           std::regex::icase);
    std::smatch match;

    if (std::regex_match(value, match, mul_pattern))
    {
        double c = std::stod(match[1].str());
        return c * dealii::numbers::PI;
    }

    // Otherwise, just parse numeric literal
    return std::stod(value);
}

std::map<std::string, double>
parse_constants_with_pi_and_multiplication(const std::string& s)
{
    std::map<std::string, double> m;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        auto pos = item.find('=');
        if (pos == std::string::npos)
            continue;

        std::string key = item.substr(0, pos);
        std::string value = item.substr(pos + 1);

        auto trim = [](std::string& x) {
            x.erase(0, x.find_first_not_of(" \t"));
            x.erase(x.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(value);

        m[key] = parse_value_with_pi(value);
    }

    return m;
}
