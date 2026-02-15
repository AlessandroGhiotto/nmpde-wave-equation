#include "WaveTheta.hpp"

void WaveTheta::setup()
{
    pcout << "===============================================" << std::endl;

    // Use base class methods
    setup_mesh();
    pcout << "-----------------------------------------------" << std::endl;

    setup_fe();
    pcout << "-----------------------------------------------" << std::endl;

    setup_dof_handler();
    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system
    {
        pcout << "Initializing the linear system" << std::endl;

        const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

        pcout << "  Initializing the sparsity pattern" << std::endl;
        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);
        sparsity.compress();

        pcout << "  Initializing matrices" << std::endl;
        mass_matrix.reinit(sparsity);
        stiffness_matrix.reinit(sparsity);
        matrix_u.reinit(sparsity);
        matrix_v.reinit(sparsity);

        pcout << "  Initializing vectors" << std::endl;
        solution_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        pcout << "Setup complete!" << std::endl;
    }
}

void WaveTheta::assemble_matrices()
{
    pcout << "Assembling mass and stiffness matrices" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix = 0.0;
    stiffness_matrix = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_mass = 0.0;
        cell_stiffness = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const double c_val = c.value(fe_values.quadrature_point(q));
            const double c2 = c_val * c_val;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_mass(i, j) += fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) * JxW;
                    cell_stiffness(i, j) += c2 *
                                            fe_values.shape_grad(i, q) *
                                            fe_values.shape_grad(j, q) * JxW;
                }
            }
        }

        cell->get_dof_indices(dof_indices);
        mass_matrix.add(dof_indices, cell_mass);
        stiffness_matrix.add(dof_indices, cell_stiffness);
    }

    mass_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);

    // Matrix for u linear system: M + (θ Δt)^2 K
    matrix_u.copy_from(mass_matrix);
    matrix_u.add((theta * delta_t) * (theta * delta_t), stiffness_matrix);

    // Matrix for v linear system: M
    matrix_v.copy_from(mass_matrix);

    // Build AMG preconditioners once on the clean matrices (reused every time step)
    {
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = true;
        amg_data.higher_order_elements = false;
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 0.02;
        preconditioner_u.initialize(matrix_u, amg_data);
        preconditioner_v.initialize(matrix_v, amg_data);
    }
    pcout << "  AMG preconditioners built on matrix_u and matrix_v" << std::endl;
}

void WaveTheta::assemble_rhs_u()
{
    // assembling rhs for u linear system
    // Aun+1=RHS(un,vn,fn,fn+1,θ,Δt)

    // RHS = M * u^n + dt * M * v^n - dt * (1-theta) * dt * K * u^

    // system_rhs = M * u^n
    mass_matrix.vmult(system_rhs, old_solution_u);

    // - k^2 * theta * (1-theta) * A * U^{n-1}
    TrilinosWrappers::MPI::Vector tmp(old_solution_u);
    stiffness_matrix.vmult(tmp, old_solution_u); // K * u^n
    system_rhs.add(-delta_t * delta_t * theta * (1 - theta), tmp);

    TrilinosWrappers::MPI::Vector tmp_v(old_solution_v);
    mass_matrix.vmult(tmp_v, old_solution_v); // M * v^n
    system_rhs.add(delta_t, tmp_v);

    // Forcing term
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    TrilinosWrappers::MPI::Vector rhs_f(system_rhs);
    rhs_f = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const Point<dim>& x_q = fe_values.quadrature_point(q);

            // Forcing term: f^n at t^n, f^{n+1} at t^{n+1}
            // Note: time has already been incremented to t^{n+1} in run()
            f.set_time(time - delta_t);
            const double f_n = f.value(x_q);

            f.set_time(time);
            const double f_np1 = f.value(x_q);

            const double f_avg = theta * f_np1 + (1.0 - theta) * f_n;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += theta * delta_t * delta_t * f_avg * fe_values.shape_value(i, q) * JxW;
        }

        cell->get_dof_indices(dof_indices);
        rhs_f.add(dof_indices, cell_rhs);
    }

    system_rhs.add(1.0, rhs_f);
    system_rhs.compress(VectorOperation::add);

    // Dirichlet boundary conditions moved to solve_u to avoid modifying matrix_u
}

void WaveTheta::assemble_rhs_v()
{
    // RHS = M*v^n
    mass_matrix.vmult(system_rhs, old_solution_v);

    // RHS -= dt*(1-theta)*K*u^n
    TrilinosWrappers::MPI::Vector tmp(old_solution_u);
    stiffness_matrix.vmult(tmp, old_solution_u);
    system_rhs.add(-delta_t * (1.0 - theta), tmp);

    TrilinosWrappers::MPI::Vector tmp2(solution_u);
    stiffness_matrix.vmult(tmp2, solution_u);
    system_rhs.add(-delta_t * theta, tmp2);

    // Forcing term
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    TrilinosWrappers::MPI::Vector rhs_f(system_rhs);
    rhs_f = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const Point<dim>& x_q = fe_values.quadrature_point(q);

            // Forcing term: f^n at t^n, f^{n+1} at t^{n+1}
            // Note: time has already been incremented to t^{n+1} in run()
            f.set_time(time - delta_t);
            const double f_n = f.value(x_q);

            f.set_time(time);
            const double f_np1 = f.value(x_q);

            const double f_avg = theta * f_np1 + (1.0 - theta) * f_n;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += delta_t * f_avg * fe_values.shape_value(i, q) * JxW;
        }

        cell->get_dof_indices(dof_indices);
        rhs_f.add(dof_indices, cell_rhs);
    }

    system_rhs.add(1.0, rhs_f);
    system_rhs.compress(VectorOperation::add);

    // Boundary condition moved to solve_v to avoid modifying matrix_v
}

void WaveTheta::solve_u()
{
    // Create a temporary system matrix to apply BCs without destroying the global matrix_u
    TrilinosWrappers::SparseMatrix system_matrix;
    system_matrix.reinit(matrix_u);
    system_matrix.copy_from(matrix_u);

    // Dirichlet boundary conditions at t^{n+1}
    // Note: time has already been incremented to t^{n+1} in run()
    {
        g.set_time(time);

        std::map<types::global_dof_index, double> boundary_values_u;
        std::map<types::boundary_id, const Function<dim>*> boundary_functions_u;

        for (const auto id : mesh.get_boundary_ids())
            boundary_functions_u[id] = &g;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions_u,
                                                 boundary_values_u);

        MatrixTools::apply_boundary_values(
            boundary_values_u, system_matrix, solution_u, system_rhs, true);
    }

    ReductionControl solver_control(10000, 1e-12, 1e-6);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    // Use cached AMG preconditioner (built once in assemble_matrices)
    solver.solve(system_matrix, solution_u, system_rhs, preconditioner_u);

    current_iterations_u = solver_control.last_step();
}

void WaveTheta::solve_v()
{
    // Create a temporary system matrix to apply BCs without destroying the global matrix_v
    TrilinosWrappers::SparseMatrix system_matrix;
    system_matrix.reinit(matrix_v);
    system_matrix.copy_from(matrix_v);

    // Boundary condition at t^{n+1}
    // Note: time has already been incremented to t^{n+1} in run()
    {
        dgdt.set_time(time);

        std::map<types::global_dof_index, double> boundary_values_v;
        std::map<types::boundary_id, const Function<dim>*> boundary_functions_v;

        for (const auto id : mesh.get_boundary_ids())
            boundary_functions_v[id] = &dgdt;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions_v,
                                                 boundary_values_v);

        MatrixTools::apply_boundary_values(
            boundary_values_v, system_matrix, solution_v, system_rhs, true);
    }

    ReductionControl solver_control(10000, 1e-12, 1e-6);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    // Use cached AMG preconditioner (built once in assemble_matrices)
    solver.solve(system_matrix, solution_v, system_rhs, preconditioner_v);

    current_iterations_v = solver_control.last_step();
}

void WaveTheta::run()
{
    setup();
    assemble_matrices();

    // Call base class method with theta-specific parameters
    std::string method_params = "-theta" + clean_double(theta);
    prepare_output_filename(method_params);

    pcout << "Setting initial conditions..." << std::endl;

    VectorTools::interpolate(dof_handler, u0, old_solution_u);
    VectorTools::interpolate(dof_handler, v0, old_solution_v);

    solution_u = old_solution_u;
    solution_v = old_solution_v;

    pcout << "||u0|| = " << old_solution_u.l2_norm() << std::endl;
    pcout << "||v0|| = " << old_solution_v.l2_norm() << std::endl;
    pcout << "-----------------------------------------------" << std::endl;

    output();
    timestep_number = 0;
    time = 0.0;
    const double divergence_threshold = 1e130;
    unsigned int total_iterations_u = 0;
    unsigned int total_iterations_v = 0;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    while (time < T)
    {
        time += delta_t;
        ++timestep_number;

        assemble_rhs_u();
        solve_u();
        total_iterations_u += current_iterations_u;

        assemble_rhs_v();
        solve_v();
        total_iterations_v += current_iterations_v;

        const double norm_u = solution_u.l2_norm();
        const double norm_v = solution_v.l2_norm();
        if (check_divergence(norm_u, norm_v, divergence_threshold))
        {
            pcout << "Divergence detected at step " << timestep_number
                  << ", t = " << time << "; stopping simulation." << std::endl;
            break;
        }

        old_solution_u = solution_u;
        old_solution_v = solution_v;

        if (log_every > 0 && (timestep_number % log_every == 0))
        {
            compute_and_log_energy();
            compute_and_log_error();
            log_iterations(current_iterations_u, current_iterations_v);
        }

        if (timestep_number % print_every == 0)
        {
            print_step_info();
        }

        output();
    }

    // Stop timer and compute elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    simulation_time = std::chrono::duration<double>(end_time - start_time).count();

    pcout << "\nSimulation completed: " << timestep_number
          << " steps, final time t = " << time << std::endl;
    pcout << "Elapsed time: " << std::fixed << std::setprecision(3)
          << simulation_time << " seconds" << std::endl;
    pcout << "Total CG iterations (u): " << total_iterations_u
          << ", avg per step: " << std::fixed << std::setprecision(1)
          << (timestep_number > 0 ? static_cast<double>(total_iterations_u) / timestep_number : 0.0)
          << std::endl;
    pcout << "Total CG iterations (v): " << total_iterations_v
          << ", avg per step: " << std::fixed << std::setprecision(1)
          << (timestep_number > 0 ? static_cast<double>(total_iterations_v) / timestep_number : 0.0)
          << std::endl;

    // Compute final errors with logging of theta
    compute_final_errors(std::to_string(theta), "", "");

    // Close log files
    if (mpi_rank == 0)
    {
        if (energy_log_file.is_open())
            energy_log_file.close();
        if (error_log_file.is_open())
            error_log_file.close();
        if (convergence_file.is_open())
            convergence_file.close();
        if (iterations_log_file.is_open())
            iterations_log_file.close();
    }
}
