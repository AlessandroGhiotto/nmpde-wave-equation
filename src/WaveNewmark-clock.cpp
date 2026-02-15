#include "WaveNewmark.hpp"
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TimeMonitor.hpp>

void WaveNewmark::setup()
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
        matrix_a.reinit(sparsity);
        system_matrix_a.reinit(sparsity);

        pcout << "  Initializing vectors" << std::endl;
        solution_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_a.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_a.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        pcout << "  Setup complete!" << std::endl;
    }
}

void WaveNewmark::assemble_matrices()
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

    // Matrix for a linear system: M + Δt^2 β A
    matrix_a.copy_from(mass_matrix);
    matrix_a.add(beta * delta_t * delta_t, stiffness_matrix);
    // AMG preconditioner will be built on first solve (on BC-modified matrix)
}

void WaveNewmark::assemble_rhs()
{
    // RHS for (M + dt^2 * beta * A) a^{n+1} = f^{n+1} - A z
    // where z = u^n + dt v^n + dt^2 (0.5 - beta) a^n

    system_rhs = 0.0;

    // --- Local vector arithmetic (no MPI) ---
    TrilinosWrappers::MPI::Vector z(old_solution_u);
    {
        Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("rhs:vector_ops"));
        z.add(delta_t, old_solution_v);
        z.add(delta_t * delta_t * (0.5 - beta), old_solution_a);
    }

    // --- SpMV: involves ghost exchange (MPI communication) ---
    TrilinosWrappers::MPI::Vector w(system_rhs);
    {
        Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("rhs:vmult"));
        stiffness_matrix.vmult(w, z);
    }

    // RHS = -w  (local)
    system_rhs.add(-1.0, w);

    // --- Local cell loop for forcing term ---
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    TrilinosWrappers::MPI::Vector rhs_f(system_rhs);
    rhs_f = 0.0;

    f.set_time(time);

    {
        Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("rhs:cell_loop"));
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
                const double f_np1 = f.value(x_q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += f_np1 * fe_values.shape_value(i, q) * JxW;
            }

            cell->get_dof_indices(dof_indices);
            rhs_f.add(dof_indices, cell_rhs);
        }
    }

    system_rhs.add(1.0, rhs_f);

    // --- compress: MPI communication (Allreduce / exchange) ---
    {
        Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("rhs:compress"));
        system_rhs.compress(VectorOperation::add);
    }
}

void WaveNewmark::solve_a()
{
    // --- Matrix copy + BC application (local, no MPI) ---
    {
        Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("solve:BC_setup"));

        system_matrix_a.copy_from(matrix_a);

        std::map<types::global_dof_index, double> boundary_values_a;
        std::map<types::global_dof_index, double> boundary_values_u;

        g.set_time(time);

        for (const auto id : mesh.get_boundary_ids())
            VectorTools::interpolate_boundary_values(dof_handler, id, g, boundary_values_u);

        const double beta_dt2 = beta * delta_t * delta_t;

        if (beta > 1e-12)
        {
            for (const auto& bv : boundary_values_u)
            {
                const types::global_dof_index dof = bv.first;
                const double g_val = bv.second;

                const double u_pred = old_solution_u(dof) +
                                      delta_t * old_solution_v(dof) +
                                      delta_t * delta_t * (0.5 - beta) * old_solution_a(dof);

                boundary_values_a[dof] = (g_val - u_pred) / beta_dt2;
            }
        }

        MatrixTools::apply_boundary_values(boundary_values_a, system_matrix_a,
                                           solution_a, system_rhs, true);
    }

    // Build AMG on the first call (on the actual BC-modified matrix)
    if (!preconditioner_a_initialized)
    {
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = true;
        amg_data.higher_order_elements = false;
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 0.02;
        preconditioner_a.initialize(system_matrix_a, amg_data);
        preconditioner_a_initialized = true;
        pcout << "  AMG preconditioner built on BC-modified matrix_a" << std::endl;
    }

    // --- CG solve: contains MPI communication (dot products = Allreduce, SpMV = ghost exchange) ---
    ReductionControl solver_control(10000, 1e-12, 1e-6);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    {
        Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("solve:CG"));
        solver.solve(system_matrix_a, solution_a, system_rhs, preconditioner_a);
    }

    current_iterations = solver_control.last_step();
}

void WaveNewmark::update_u_v()
{
    // Newmark updates (purely local vector arithmetic, no MPI)
    Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("update:vector_ops"));

    solution_u = old_solution_u;
    solution_u.add(delta_t, old_solution_v);
    solution_u.add(delta_t * delta_t * (0.5 - beta), old_solution_a);
    solution_u.add(delta_t * delta_t * beta, solution_a);

    solution_v = old_solution_v;
    solution_v.add(delta_t * (1.0 - gamma), old_solution_a);
    solution_v.add(delta_t * gamma, solution_a);
}

void WaveNewmark::run()
{
    setup();
    assemble_matrices();

    // Call base class method with Newmark-specific parameters
    std::string method_params = "-gamma" + clean_double(gamma) +
                                "-beta" + clean_double(beta);
    prepare_output_filename(method_params);

    pcout << "Setting initial conditions..." << std::endl;

    VectorTools::interpolate(dof_handler, u0, old_solution_u);
    VectorTools::interpolate(dof_handler, v0, old_solution_v);

    solution_u = old_solution_u;
    solution_v = old_solution_v;

    // Compute consistent initial acceleration a^0 by solving:
    //   M a^0 = f(0) - K u^0
    // Setting a^0 = 0 would introduce an O(1) error that degrades convergence.
    {
        pcout << "Computing consistent initial acceleration a^0..." << std::endl;

        // rhs_a = -K * u^0
        TrilinosWrappers::MPI::Vector rhs_a(old_solution_u);
        stiffness_matrix.vmult(rhs_a, old_solution_u);
        rhs_a *= -1.0;

        // Add forcing term f(0)
        f.set_time(0.0);
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q = quadrature->size();

        FEValues<dim> fe_values(*fe, *quadrature,
                                update_values | update_quadrature_points | update_JxW_values);

        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

        TrilinosWrappers::MPI::Vector f_vec(rhs_a);
        f_vec = 0.0;

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            fe_values.reinit(cell);
            cell_rhs = 0.0;

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double JxW = fe_values.JxW(q);
                const double f_val = f.value(fe_values.quadrature_point(q));

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += f_val * fe_values.shape_value(i, q) * JxW;
            }

            cell->get_dof_indices(dof_indices);
            f_vec.add(dof_indices, cell_rhs);
        }
        f_vec.compress(VectorOperation::add);
        rhs_a.add(1.0, f_vec);

        // Solve M a^0 = f(0) - K u^0  (one-time solve, use simple SSOR)
        TrilinosWrappers::PreconditionSSOR preconditioner;
        preconditioner.initialize(
            mass_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

        ReductionControl solver_control(10000, 1e-12, 1e-6);
        SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

        solver.solve(mass_matrix, old_solution_a, rhs_a, preconditioner);
        solution_a = old_solution_a;

        pcout << "  ||a0|| = " << old_solution_a.l2_norm()
              << " (" << solver_control.last_step() << " CG iterations)" << std::endl;
    }

    pcout << "||u0|| = " << old_solution_u.l2_norm() << std::endl;
    pcout << "||v0|| = " << old_solution_v.l2_norm() << std::endl;
    pcout << "-----------------------------------------------" << std::endl;

    output();
    timestep_number = 0;
    time = 0.0;
    const double divergence_threshold = 1e130;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    auto stacked = Teuchos::rcp(new Teuchos::StackedTimer("WaveNewmark"));
    Teuchos::TimeMonitor::setStackedTimer(stacked);

    while (time < T)
    {
        time += delta_t;
        ++timestep_number;

        {
            Teuchos::TimeMonitor t_rhs(*Teuchos::TimeMonitor::getNewTimer("rhs"));
            assemble_rhs();
        }
        {
            Teuchos::TimeMonitor t_solve(*Teuchos::TimeMonitor::getNewTimer("solve"));
            solve_a(); // this also handle the BCs
        }
        {
            Teuchos::TimeMonitor t_update(*Teuchos::TimeMonitor::getNewTimer("update"));
            update_u_v();
        }
        // --- Norms: MPI Allreduce ---
        double norm_u, norm_v;
        {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("norms"));
            norm_u = solution_u.l2_norm();
            norm_v = solution_v.l2_norm();
        }
        if (check_divergence(norm_u, norm_v, divergence_threshold))
        {
            pcout << "Divergence detected at step " << timestep_number
                  << ", t = " << time << "; stopping simulation." << std::endl;
            break;
        }

        // --- Vector copies (local, no MPI) ---
        {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("copy_old"));
            old_solution_u = solution_u;
            old_solution_v = solution_v;
            old_solution_a = solution_a;
        }

        if (log_every > 0 && (timestep_number % log_every == 0))
        {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("logging"));
            compute_and_log_energy();
            compute_and_log_error();
            log_iterations(current_iterations, 0);
        }

        if (timestep_number % print_every == 0)
        {
            print_step_info();
        }

        {
            Teuchos::TimeMonitor t(*Teuchos::TimeMonitor::getNewTimer("output"));
            output();
        }
    }

    // Stop timer and compute elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    simulation_time = std::chrono::duration<double>(end_time - start_time).count();

    pcout << "\nSimulation completed: " << timestep_number
          << " steps, final time t = " << time << std::endl;
    pcout << "Elapsed time: " << std::fixed << std::setprecision(3)
          << simulation_time << " seconds" << std::endl;

    // Compute final errors with Newmark parameters to be logged in csv
    compute_final_errors("", std::to_string(beta), std::to_string(gamma));

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

    stacked->report(std::cout, Teuchos::DefaultComm<int>::getComm());
    Teuchos::TimeMonitor::summarize();
}
