/* ============================================================
 * WaveEquationBase.tpp  —  included by WaveEquationBase.hpp
 * ============================================================ */

#pragma once

#include <filesystem>

namespace WaveSolver {
  using namespace dealii;

  // ── Constructor ────────────────────────────────────────────
  template <int dim>
  WaveEquationBase<dim>::WaveEquationBase(const Parameters &prm)
    : prm(prm),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      mpi_comm(MPI_COMM_WORLD),
      triangulation(mpi_comm),
      fe(prm.fe_degree),
      dof_handler(triangulation),
      time(0.), dt(prm.dt), step_number(0)
  {}

  // ── Grid ───────────────────────────────────────────────────
  template <int dim>
  void WaveEquationBase<dim>::make_grid() {
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(prm.refinements);
  }

  // ── DoF setup ──────────────────────────────────────────────
  template <int dim>
  void WaveEquationBase<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned_dofs, mpi_comm, locally_relevant_dofs);

    mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);
    stiffness_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

    solution.reinit(locally_owned_dofs, mpi_comm);
    velocity.reinit(locally_owned_dofs, mpi_comm);
    acceleration.reinit(locally_owned_dofs, mpi_comm);
  }

  // ── Matrix assembly ────────────────────────────────────────
  template <int dim>
  void WaveEquationBase<dim>::assemble_matrices() {
    QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature,
                            update_values | update_gradients | update_JxW_values);
    const unsigned int dpc = fe.n_dofs_per_cell();
    const unsigned int nq  = quadrature.size();

    FullMatrix<double> cell_mass(dpc, dpc);
    FullMatrix<double> cell_stiff(dpc, dpc);
    std::vector<types::global_dof_index> dof_indices(dpc);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned()) continue;
      fe_values.reinit(cell);
      cell_mass = 0.; cell_stiff = 0.;
      for (unsigned int q = 0; q < nq; ++q) {
        const double JxW = fe_values.JxW(q);
        for (unsigned int i = 0; i < dpc; ++i)
          for (unsigned int j = 0; j < dpc; ++j) {
            cell_mass(i,j)  += fe_values.shape_value(i,q)*fe_values.shape_value(j,q)*JxW;
            cell_stiff(i,j) += fe_values.shape_grad(i,q) *fe_values.shape_grad(j,q) *JxW;
          }
      }
      cell->get_dof_indices(dof_indices);
      constraints.distribute_local_to_global(cell_mass,  dof_indices, mass_matrix);
      constraints.distribute_local_to_global(cell_stiff, dof_indices, stiffness_matrix);
    }
    mass_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);
  }

  // ── Dirichlet BC ───────────────────────────────────────────
  template <int dim>
  void WaveEquationBase<dim>::apply_dirichlet_bc(
    TrilinosWrappers::MPI::Vector &v, double t) const
  {
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(
      dof_handler, 0,
      WaveExact::ExactSolution<dim>(prm.wave_speed, t), bv);

    TrilinosWrappers::MPI::Vector result(locally_owned_dofs, mpi_comm);
    for (const auto idx : locally_owned_dofs) {
      auto it = bv.find(idx);
      result[idx] = (it != bv.end()) ? it->second : v[idx];
    }
    result.compress(VectorOperation::insert);
    v = result;
  }

  // ── Acceleration solve:  M*acc = -c^2 * K * u ─────────────
  template <int dim>
  void WaveEquationBase<dim>::compute_acceleration(
    TrilinosWrappers::MPI::Vector &acc,
    const TrilinosWrappers::MPI::Vector &u) const
  {
    TrilinosWrappers::MPI::Vector rhs(locally_owned_dofs, mpi_comm);
    stiffness_matrix.vmult(rhs, u);
    rhs *= -(prm.wave_speed * prm.wave_speed);

    acc = 0.;
    SolverControl sc(2000, 1e-12 * rhs.l2_norm() + 1e-14);
    TrilinosWrappers::SolverCG cg(sc);
    TrilinosWrappers::PreconditionJacobi prec;
    prec.initialize(mass_matrix);
    cg.solve(mass_matrix, acc, rhs, prec);
    constraints.distribute(acc);
  }

  // ── Discrete energy  E = ½(v^T M v + c^2 u^T K u) ────────
  template <int dim>
  double WaveEquationBase<dim>::compute_energy() const {
    TrilinosWrappers::MPI::Vector gu(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    TrilinosWrappers::MPI::Vector gv(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    gu = solution; gv = velocity;

    QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> fev(fe, quad, update_values | update_gradients | update_JxW_values);
    const unsigned int nq = quad.size();
    std::vector<double>        u_val(nq), v_val(nq);
    std::vector<Tensor<1,dim>> u_grad(nq);

    double local_e = 0.;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned()) continue;
      fev.reinit(cell);
      fev.get_function_values   (gu, u_val);
      fev.get_function_values   (gv, v_val);
      fev.get_function_gradients(gu, u_grad);
      for (unsigned int q = 0; q < nq; ++q) {
        double grad_sq = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          grad_sq += u_grad[q][d]*u_grad[q][d];
        local_e += 0.5*(v_val[q]*v_val[q]
                  + prm.wave_speed*prm.wave_speed*grad_sq) * fev.JxW(q);
      }
    }
    return Utilities::MPI::sum(local_e, mpi_comm);
  }

  // ── L2 error against exact solution ───────────────────────
  template <int dim>
  double WaveEquationBase<dim>::compute_l2_error(double t) const {
    TrilinosWrappers::MPI::Vector gu(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    gu = solution;
    Vector<double> cell_err(triangulation.n_active_cells());
    VectorTools::integrate_difference(
      dof_handler, gu,
      WaveExact::ExactSolution<dim>(prm.wave_speed, t),
      cell_err, QGauss<dim>(fe.degree + 2), VectorTools::L2_norm);
    return VectorTools::compute_global_error(
      triangulation, cell_err, VectorTools::L2_norm);
  }

  // ── VTU output ─────────────────────────────────────────────
  template <int dim>
  void WaveEquationBase<dim>::output_vtu(unsigned int step) const {
    TrilinosWrappers::MPI::Vector gu(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    TrilinosWrappers::MPI::Vector gv(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    gu = solution; gv = velocity;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(gu, "displacement");
    data_out.add_data_vector(gv, "velocity");

    TrilinosWrappers::MPI::Vector exact_owned(locally_owned_dofs, mpi_comm);
    VectorTools::interpolate(dof_handler,
      WaveExact::ExactSolution<dim>(prm.wave_speed, time), exact_owned);
    TrilinosWrappers::MPI::Vector exact_vec(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    exact_vec = exact_owned;
    data_out.add_data_vector(exact_vec, "exact_solution");

    // Pointwise error field for ParaView.
    // Must compute on an owned (non-ghosted) vector — ghosted vectors are read-only.
    TrilinosWrappers::MPI::Vector err_owned(locally_owned_dofs, mpi_comm);
    err_owned = exact_owned;                     // exact - u
    err_owned.add(-1.0, solution);
    TrilinosWrappers::MPI::Vector err_vec(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    err_vec = err_owned;
    data_out.add_data_vector(err_vec, "error");

    data_out.build_patches(fe.degree);
    data_out.write_vtu_with_pvtu_record(
      prm.output_dir + "/", "wave", step, mpi_comm, 4);
  }

  // ── Main run loop ──────────────────────────────────────────
  template <int dim>
  void WaveEquationBase<dim>::run() {
    make_grid();
    setup_system();
    assemble_matrices();

    // Print mesh / CFL info
    const double h   = 1.0 / std::pow(2.0, (double)prm.refinements);
    const double cfl = prm.wave_speed * dt * std::sqrt((double)dim) / h;
    pcout << "─────────────────────────────────────────\n"
          << " Scheme:    " << scheme_name() << "\n"
          << " DoFs:      " << dof_handler.n_dofs() << "\n"
          << " h:         " << h    << "\n"
          << " dt:        " << dt   << "\n"
          << " CFL:       " << std::fixed << std::setprecision(3) << cfl << "\n"
          << "─────────────────────────────────────────\n";

    // ── Initial conditions ──────────────────────────────────
    VectorTools::interpolate(dof_handler,
      WaveExact::ExactSolution<dim>(prm.wave_speed, 0.), solution);
    constraints.distribute(solution);
    VectorTools::interpolate(dof_handler,
      WaveExact::InitialVelocity<dim>(), velocity);
    constraints.distribute(velocity);

    // Bootstrap acceleration  a^0 = M^{-1}(-c^2 K u^0)
    compute_acceleration(acceleration, solution);

    // Scheme-specific initialisation (e.g. Leapfrog sets u^{-1})
    init_scheme_state();

    // ── Open log files ──────────────────────────────────────
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    if (rank == 0) {
      // Create output directory if it doesn't exist
      std::filesystem::create_directories(prm.output_dir);
      energy_log.open(prm.output_dir + "/energy.csv");
      energy_log << "step,time,energy,energy_ratio\n";
      error_log.open(prm.output_dir + "/error.csv");
      error_log << "step,time,l2_error\n";
    }

    const double E0 = compute_energy();
    pcout << std::fixed << std::setprecision(4)
          << "t = " << time
          << "  E = " << E0
          << "  L2 = " << compute_l2_error(0.) << "\n";
    if (prm.output_every > 0) output_vtu(0);

    // ── Time loop ───────────────────────────────────────────
    Timer timer;
    double total_time = 0.;
    const unsigned int n_steps =
      static_cast<unsigned int>(std::round(prm.final_time / dt));
    const unsigned int print_every = std::max(1u, n_steps / 5);
    const unsigned int log_every   = std::max(1u, n_steps / 20);

    for (step_number = 1; step_number <= n_steps; ++step_number) {
      time = step_number * dt;
      timer.restart();
      advance_one_step();
      total_time += timer.wall_time();

      const double E = compute_energy();

      if (step_number % print_every == 0)
        pcout << "t = " << std::fixed    << std::setprecision(4) << time
              << "  E/E0 = " << std::scientific << E / E0
              << "  L2 = "   << compute_l2_error(time) << "\n";

      if (step_number % log_every == 0) {
        const double log_err = compute_l2_error(time);   // collective MPI
        if (rank == 0)
          error_log << step_number << "," << time << "," << log_err << "\n";
      }
      if (rank == 0)
        energy_log << step_number << "," << time << "," << E << "," << E/E0 << "\n";

      if (prm.output_every > 0 && step_number % prm.output_every == 0)
        output_vtu(step_number);
    }

    // ── Summary ─────────────────────────────────────────────
    const double final_err = compute_l2_error(prm.final_time);
    const double final_er  = compute_energy() / E0;

    pcout << "\nFinal L2 error: " << std::scientific << std::setprecision(6) << final_err << "\n"
          << "Final E/E0: "       << std::scientific << std::setprecision(6) << final_er  << "\n"
          << "Wall time: "        << std::fixed      << std::setprecision(3) << total_time << "\n";

    if (rank == 0) {
      std::ofstream cv(prm.output_dir + "/convergence.csv", std::ios::app);
      cv.seekp(0, std::ios::end);
      if (cv.tellp() == 0)
        cv << "scheme,fe_degree,refinements,h,dt,dofs,final_l2_error,final_energy_ratio,wall_time\n";
      cv << scheme_name() << ","
         << prm.fe_degree << ","
         << prm.refinements << ","
         << h << ","
         << dt << ","
         << dof_handler.n_dofs() << ","
         << final_err << ","
         << final_er  << ","
         << total_time << "\n";
    }

    energy_log.close();
    error_log.close();
  }

} // namespace WaveSolver
