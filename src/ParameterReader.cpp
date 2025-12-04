#include "ParameterReader.hpp"
#include "wave.hpp"

static constexpr unsigned int dim = Wave::dim;

double parse_value_with_pi(std::string value);
std::map<std::string, double>
parse_constants_with_pi_and_multiplication(const std::string& s);

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

// ------------------
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
