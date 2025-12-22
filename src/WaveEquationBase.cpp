#include "WaveEquationBase.hpp"
#include <cstdlib> // getenv

namespace
{
bool env_flag_enabled(const char* name, const bool default_value)
{
    const char* v = std::getenv(name);
    if (!v)
        return default_value;
    const std::string s(v);
    if (s == "0" || s == "false" || s == "FALSE" || s == "False")
        return false;
    if (s == "1" || s == "true" || s == "TRUE" || s == "True")
        return true;
    return default_value;
}
} // namespace

void WaveEquationBase::setup_mesh()
{
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;
    GridGenerator::subdivided_hyper_rectangle_with_simplices(
        mesh_serial,
        { N_el.first, N_el.second },
        geometry.first, geometry.second,
        false);

    if (mpi_rank == 0)
    {
        if (!std::filesystem::exists("../mesh/"))
            std::filesystem::create_directories("../mesh/");

        const std::string mesh_file_name =
            "../mesh/rectangle-simplices-" + std::to_string(N_el.first) + "x" +
            std::to_string(N_el.second) + "-" + clean_double(geometry.first[0], 2) +
            "_" + clean_double(geometry.second[0], 2) + "x" +
            clean_double(geometry.first[1], 2) + "_" + clean_double(geometry.second[1], 2) + ".vtk";

        GridOut grid_out;
        std::ofstream grid_out_file(mesh_file_name);
        grid_out.write_vtk(mesh_serial, grid_out_file);
        pcout << "  Mesh saved to " << mesh_file_name << std::endl;
    }

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data =
        TriangulationDescription::Utilities::create_description_from_triangulation(
            mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
}

void WaveEquationBase::setup_fe()
{
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);
    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
}

void WaveEquationBase::setup_dof_handler()
{
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
}

void WaveEquationBase::prepare_output_filename(const std::string& method_params)
{
    output_folder = "../results/" + problem_name + "/run-R" + std::to_string(r) +
                    "-N" + std::to_string(N_el.first) + "x" + std::to_string(N_el.second) +
                    "-dt" + clean_double(delta_t) + "-T" + clean_double(T) + method_params + "/";

    if (mpi_rank == 0)
    {
        if (!std::filesystem::exists(output_folder))
            std::filesystem::create_directories(output_folder);

        // Copy the parameter file for reproducibility, if provided.
        if (const char* param_env = std::getenv("NMPDE_PARAM_FILE"))
        {
            try
            {
                const std::filesystem::path src(param_env);
                const std::filesystem::path dst = std::filesystem::path(output_folder) / "parameters.json";
                if (std::filesystem::exists(src))
                {
                    std::filesystem::copy_file(src, dst,
                                               std::filesystem::copy_options::overwrite_existing);
                    pcout << "  Parameters copied to " << dst << std::endl;
                }
                else
                {
                    pcout << "  Parameter file not found: " << src << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                pcout << "  Warning: could not copy parameter file (" << e.what() << ")" << std::endl;
            }
        }

        // NOTE: energy/error CSV files are opened lazily in compute_and_log_*()
        // so log_every=0 produces no files.

        if (exact_solution != nullptr)
        {
            // Keep convergence file independent of time-series logging.
            std::string convergence_file_path = "../results/" + problem_name + "/convergence.csv";
            bool file_exists = std::filesystem::exists(convergence_file_path);
            convergence_file.open(convergence_file_path, std::ios_base::app);
            if (convergence_file.is_open() && !file_exists)
                convergence_file << "h,N_el_x,N_el_y,r,dt,T,method,theta,beta,gamma,rel_L2_error_final,rel_H1_error_final,elapsed_time_s" << std::endl;
        }
    }
}

void WaveEquationBase::compute_and_log_energy()
{
    TrilinosWrappers::MPI::Vector tmp_u(solution_u);
    TrilinosWrappers::MPI::Vector tmp_v(solution_v);
    stiffness_matrix.vmult(tmp_u, solution_u);
    mass_matrix.vmult(tmp_v, solution_v);
    current_energy = 0.5 * (tmp_v * solution_v + tmp_u * solution_u);

    if (mpi_rank == 0)
    {
        if (!energy_log_file.is_open())
        {
            energy_log_file.open(output_folder + "energy.csv");
            if (energy_log_file.is_open())
                energy_log_file << "timestep,time,energy" << std::endl;
        }

        if (energy_log_file.is_open())
            energy_log_file << timestep_number << "," << time << "," << current_energy << std::endl;
    }
}

void WaveEquationBase::compute_and_log_error()
{
    if (exact_solution == nullptr)
        return;

    exact_solution->set_time(time);

    const double error_L2 = compute_error(VectorTools::L2_norm, *exact_solution);
    const double error_H1 = compute_error(VectorTools::H1_norm, *exact_solution);
    const double rel_error_L2 = compute_relative_error(error_L2, VectorTools::L2_norm, *exact_solution);
    const double rel_error_H1 = compute_relative_error(error_H1, VectorTools::H1_norm, *exact_solution);

    if (mpi_rank == 0)
    {
        if (!error_log_file.is_open())
        {
            error_log_file.open(output_folder + "error.csv");
            if (error_log_file.is_open())
                error_log_file << "timestep,time,L2_error,H1_error,rel_L2_error,rel_H1_error" << std::endl;
        }

        if (error_log_file.is_open())
        {
            error_log_file << timestep_number << "," << time << ","
                           << std::scientific << std::setprecision(6)
                           << error_L2 << "," << error_H1 << ","
                           << rel_error_L2 << "," << rel_error_H1 << std::endl;
        }
    }

    // NOTE: no accumulation/averaging; final error computed at the end.
}

void WaveEquationBase::compute_final_errors()
{
    compute_final_errors("", "", "");
}

void WaveEquationBase::compute_final_errors(const std::string& theta_str,
                                            const std::string& beta_str,
                                            const std::string& gamma_str)
{
    if (exact_solution == nullptr)
        return;

    // Recompute the error at the final (current) time using the final (current) solution.
    exact_solution->set_time(time);

    const double error_L2 = compute_error(VectorTools::L2_norm, *exact_solution);
    const double error_H1 = compute_error(VectorTools::H1_norm, *exact_solution);
    const double rel_error_L2 = compute_relative_error(error_L2, VectorTools::L2_norm, *exact_solution);
    const double rel_error_H1 = compute_relative_error(error_H1, VectorTools::H1_norm, *exact_solution);

    if (mpi_rank == 0 && convergence_file.is_open())
    {
        const double h = 1.0 / std::sqrt(N_el.first * N_el.second);
        convergence_file << h << "," << N_el.first << "," << N_el.second << "," << r << ","
                         << delta_t << "," << T << "," << problem_name << ",";
        convergence_file << (theta_str.empty() ? "N/A" : theta_str) << ","
                         << (beta_str.empty() ? "N/A" : beta_str) << ","
                         << (gamma_str.empty() ? "N/A" : gamma_str) << ",";
        convergence_file << std::scientific << std::setprecision(6)
                         << rel_error_L2 << "," << rel_error_H1 << ",";
        convergence_file << std::fixed << std::setprecision(3)
                         << simulation_time << std::endl;

        pcout << "Final (last-iteration) errors:" << std::endl;
        pcout << "  Relative L2 error  = "
              << std::scientific << std::setprecision(6) << rel_error_L2 << std::endl;
        pcout << "  Relative H1 error  = "
              << std::scientific << std::setprecision(6) << rel_error_H1 << std::endl;
    }
}

void WaveEquationBase::print_step_info()
{
    std::ostringstream oss;
    oss << "Step " << std::setw(6) << timestep_number
        << ",  t=" << std::scientific << std::setprecision(3) << std::setw(9) << time
        << ",  ||u||=" << std::scientific << std::setprecision(3) << std::setw(9) << solution_u.l2_norm()
        << ",  ||v||=" << std::scientific << std::setprecision(3) << std::setw(9) << solution_v.l2_norm()
        << ",  E=" << std::scientific << std::setprecision(3) << std::setw(9) << current_energy;
    pcout << oss.str() << std::endl;
}

void WaveEquationBase::output() const
{
    const bool save_solution = env_flag_enabled("NMPDE_SAVE_SOLUTION", true);
    if (!save_solution)
        return;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution_u, "u");
    data_out.add_data_vector(dof_handler, solution_v, "v");

    if (exact_solution != nullptr)
    {
        exact_solution->set_time(time);

        IndexSet locally_owned = dof_handler.locally_owned_dofs();
        IndexSet locally_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

        TrilinosWrappers::MPI::Vector exact_solution_vector;
        exact_solution_vector.reinit(locally_owned, locally_relevant, MPI_COMM_WORLD);

        VectorTools::interpolate(dof_handler, *exact_solution, exact_solution_vector);
        exact_solution_vector.compress(VectorOperation::insert);

        // use the same overload that you used for u/v
        data_out.add_data_vector(dof_handler, exact_solution_vector, "u_exact");
    }

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(output_folder, "solution", timestep_number,
                                        MPI_COMM_WORLD, 4, static_cast<long int>(time));
}

double WaveEquationBase::compute_error(const VectorTools::NormType& cell_norm,
                                       const Function<dim>& exact_solution) const
{
    // Quadrature for the error integration
    const QGaussSimplex<dim> quadrature_error(r + 2);

    FE_SimplexP<dim> fe_linear(1);
    MappingFE mapping(fe_linear);

    // ghost vector to hold solution including ghost values
    TrilinosWrappers::MPI::Vector solution_ghosted;
    IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    solution_ghosted.reinit(locally_relevant_dofs,
                            mesh.get_communicator());
    solution_ghosted = solution_u; // imports ghost values

    // Cellwise error vector
    Vector<float> error_per_cell(mesh.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution_ghosted,
                                      exact_solution,
                                      error_per_cell,
                                      quadrature_error,
                                      cell_norm);

    // reduction
    const double global_error = VectorTools::compute_global_error(mesh,
                                                                  error_per_cell,
                                                                  cell_norm);

    return global_error;
}

double WaveEquationBase::compute_relative_error(const double error, // we give it as input, so we don't compute it twice
                                                const VectorTools::NormType& norm_type,
                                                const Function<dim>& exact_solution) const
{
    const QGaussSimplex<dim> quadrature_error(r + 2);
    FE_SimplexP<dim> fe_linear(1);
    MappingFE mapping(fe_linear);

    TrilinosWrappers::MPI::Vector zero_vector(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    zero_vector = 0.0;

    Vector<double> norm_per_cell(mesh.n_active_cells());
    VectorTools::integrate_difference(mapping, dof_handler, zero_vector,
                                      exact_solution, norm_per_cell,
                                      quadrature_error, norm_type);

    const double exact_norm = VectorTools::compute_global_error(mesh, norm_per_cell, norm_type);

    if (exact_norm < 1e-14)
        return error;

    return error / exact_norm;
}

bool WaveEquationBase::check_divergence(const double norm_u,
                                        const double norm_v,
                                        const double threshold) const
{
    return (!std::isfinite(norm_u) || !std::isfinite(norm_v) ||
            norm_u > threshold || norm_v > threshold);
}

std::string clean_double(double x, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << x;
    std::string s = out.str();

    // Only trim trailing zeros if there is a decimal point (i.e., we're trimming fractional zeros),
    // otherwise integers like "10" would incorrectly become "1".
    const auto dot_pos = s.find('.');
    if (dot_pos != std::string::npos)
    {
        while (!s.empty() && s.back() == '0')
            s.pop_back();
        if (!s.empty() && s.back() == '.')
            s.pop_back();
    }

    std::replace(s.begin(), s.end(), '.', '_');
    return s.empty() ? "0" : s;
}
