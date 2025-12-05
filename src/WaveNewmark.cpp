#include "WaveNewmark.hpp"

void WaveNewmark::setup()
{
    pcout << "===============================================" << std::endl;

    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        // Read serial mesh.
        Triangulation<dim> mesh_serial;

        {
            GridIn<dim> grid_in;
            grid_in.attach_triangulation(mesh_serial);

            std::ifstream mesh_file(mesh_file_name);
            grid_in.read_msh(mesh_file);
        }

        // Copy the serial mesh into the parallel one.
        {
            GridTools::partition_triangulation(mpi_size, mesh_serial);

            const auto construction_data = TriangulationDescription::Utilities::
                create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
            mesh.create_triangulation(construction_data);
        }

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        fe = std::make_unique<FE_SimplexP<dim>>(r);

        pcout << "  Degree                     = " << fe->degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
        const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler);

        pcout << "  Initializing the sparsity pattern" << std::endl;
        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                                   MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);
        sparsity.compress();

        pcout << "  Initializing matrices" << std::endl;
        mass_matrix.reinit(sparsity);      // M
        stiffness_matrix.reinit(sparsity); // A

        pcout << "  Initializing vectors" << std::endl;
        solution_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_a.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        old_solution_a.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        pcout << "Setup complete!" << std::endl;
    }
}

// Assembling mass and stiffness matrices

void WaveNewmark::assemble_matrices()
{
    pcout << "Assembling mass and stiffness matrices" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
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

    // Matrix for a linear system
    matrix_a.copy_from(mass_matrix);
    matrix_a.add(beta * delta_t * delta_t, stiffness_matrix); // M + Δt^2 beta A
}

void WaveNewmark::assemble_rhs()
{
    // Assembling RHS for (M + dt^2 * beta * A) a^{n+1} = f^{n+1} - A z
    // where z = u^n + dt v^n + dt^2 (0.5 - beta) a^n

    pcout << "Assembling RHS" << std::endl;

    system_rhs = 0.0;

    // z = u^n + dt v^n + dt^2 (0.5 - beta) a^n
    TrilinosWrappers::MPI::Vector z(old_solution_u);
    z.add(delta_t, old_solution_v);
    z.add(delta_t * delta_t * (0.5 - beta), old_solution_a);

    // w = A * z
    TrilinosWrappers::MPI::Vector w(system_rhs);
    stiffness_matrix.vmult(w, z);

    // RHS = f^{n+1} - w
    system_rhs.add(-1.0, w);

    // Assemble load vector for f^{n+1}
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    TrilinosWrappers::MPI::Vector rhs_f(system_rhs);
    rhs_f = 0.0;

    // Set time for forcing once per cell (no need to set per quadrature point)
    f.set_time(time + delta_t);

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

    system_rhs.add(1.0, rhs_f);
    system_rhs.compress(VectorOperation::add);

    pcout << "||rhs_u|| = " << system_rhs.l2_norm() << std::endl;

    // DIRICHLET BOUNDARY CONDITIONS
    // I add it here because I need the current system RHS
    // {
    //     // a: enforce a^{n+1} = g''(t^{n+1})
    //     d2gdt2.set_time(time + delta_t);

    //     std::map<types::global_dof_index, double> boundary_values;
    //     std::map<types::boundary_id, const Function<dim>*> boundary_functions;

    //     // Assign g to all boundary ids present in the mesh
    //     for (const auto id : mesh.get_boundary_ids())
    //         boundary_functions[id] = &d2gdt2;

    //     VectorTools::interpolate_boundary_values(dof_handler,
    //                                              boundary_functions,
    //                                              boundary_values);

    //     MatrixTools::apply_boundary_values(
    //         boundary_values, matrix_a, solution_a, system_rhs, true);
    // }
}

void WaveNewmark::apply_dirichlet_bc()
{
    // === Apply u = g(t) on the boundary ===
    std::map<types::global_dof_index, double> boundary_values_u;
    g.set_time(time); // your Function<dim> g(...)

    for (const auto id : mesh.get_boundary_ids())
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 id,
                                                 g,
                                                 boundary_values_u);

    // We don't care about the matrix here, we just rewrite the vector.
    TrilinosWrappers::MPI::Vector dummy_rhs;
    MatrixTools::apply_boundary_values(boundary_values_u,
                                       mass_matrix, // dummy
                                       solution_u,
                                       dummy_rhs,
                                       false); // do NOT eliminate rows

    // === Apply v = dg/dt (t) on the boundary ===
    std::map<types::global_dof_index, double> boundary_values_v;
    dgdt.set_time(time); // your Function<dim> dgdt(...)

    for (const auto id : mesh.get_boundary_ids())
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 id,
                                                 dgdt,
                                                 boundary_values_v);

    MatrixTools::apply_boundary_values(boundary_values_v,
                                       mass_matrix, // dummy
                                       solution_v,
                                       dummy_rhs,
                                       false);
}

void WaveNewmark::solve_a()
{
    pcout << "Solving for a^{n+1}" << std::endl;

    TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(matrix_a, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    ReductionControl solver_control(10000, 1e-12, 1e-6);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    solver.solve(matrix_a, solution_a, system_rhs, preconditioner);

    pcout << "  CG iterations: " << solver_control.last_step() << std::endl;
    pcout << "  ||a^{n+1}|| = " << solution_a.l2_norm() << std::endl;
}

void WaveNewmark::update_u_v()
{
    // Newmark updates:
    // u^{n+1} = u^n + dt v^n + dt^2 [(0.5 - beta) a^n + beta a^{n+1}]
    // v^{n+1} = v^n + dt [(1 - gamma) a^n + gamma a^{n+1}]

    // update u
    solution_u = old_solution_u;                                      // u^{n+1} <-- u^n
    solution_u.add(delta_t, old_solution_v);                          // + dt v^n
    solution_u.add(delta_t * delta_t * (0.5 - beta), old_solution_a); // + dt^2 (0.5 - beta) a^n
    solution_u.add(delta_t * delta_t * beta, solution_a);             // + dt^2 beta a^{n+1}

    // update v
    solution_v = old_solution_v;                             // v^{n+1} <-- v^n
    solution_v.add(delta_t * (1.0 - gamma), old_solution_a); // + dt (1 - gamma) a^n
    solution_v.add(delta_t * gamma, solution_a);             // + dt gamma a^{n+1}
}

void WaveNewmark::output() const
{
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, solution_u, "u");
    data_out.add_data_vector(dof_handler, solution_v, "v");
    data_out.add_data_vector(dof_handler, solution_a, "a");

    // Add vector for parallel partition.
    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    const std::filesystem::path mesh_path(mesh_file_name);
    const std::string output_file_name = "output-" + mesh_path.stem().string();

    data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                        /* basename = */ output_file_name,
                                        /* index = */ timestep_number,
                                        MPI_COMM_WORLD,
                                        /* n<_digits = */ 4,
                                        /* time = */ static_cast<long int>(time));
}

void WaveNewmark::run()
{
    setup();
    assemble_matrices();

    pcout << "Setting initial conditions..." << std::endl;

    VectorTools::interpolate(dof_handler, u0, old_solution_u);
    VectorTools::interpolate(dof_handler, v0, old_solution_v);

    solution_u = old_solution_u;
    solution_v = old_solution_v;

    pcout << "||u0|| = " << old_solution_u.l2_norm() << std::endl;
    pcout << "||v0|| = " << old_solution_v.l2_norm() << std::endl;

    output();
    timestep_number = 0;
    time = 0.0;

    while (time < T)
    {
        time += delta_t;
        ++timestep_number;

        pcout << "\n=== Time step " << timestep_number
              << ", t = " << time << " ===" << std::endl;

        // compute a
        assemble_rhs();
        solve_a();

        // update u, v
        update_u_v();

        // Enforce the displacement and velocity BCs
        apply_dirichlet_bc();

        old_solution_u = solution_u;
        old_solution_v = solution_v;
        old_solution_a = solution_a;

        if (timestep_number % 1 == 0)
            output();
    }

    pcout << "\nSimulation completed: " << timestep_number
          << " steps, final time t = " << time << std::endl;
}
