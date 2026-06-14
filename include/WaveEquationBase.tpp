#pragma once

#include <filesystem>

namespace WaveSolver {
  using namespace dealii;

  // ── Constructor 
  template <int dim>
  WaveEquationBase<dim>::WaveEquationBase(const Parameters &prm)
    : prm(prm),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      mpi_comm(MPI_COMM_WORLD),
      triangulation(mpi_comm),
      fe(prm.fe_degree),
      dof_handler(triangulation),
      time(0.), dt(prm.dt), step_number(0),
      timer_output(mpi_comm, pcout, TimerOutput::never,
                    TimerOutput::wall_times)
  {}

  // Grid
  template <int dim>
  void WaveEquationBase<dim>::make_grid() {
    TimerOutput::Scope t(timer_output, "01 mesh setup");
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(prm.refinements);
  }

  template <int dim>
  void WaveEquationBase<dim>::setup_system() {
    TimerOutput::Scope t(timer_output, "01 mesh setup");

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

  template <int dim>
  void WaveEquationBase<dim>::assemble_matrices() {
    TimerOutput::Scope t(timer_output, "02 assembly");

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

  template <int dim>
  void WaveEquationBase<dim>::apply_dirichlet_bc(
    TrilinosWrappers::MPI::Vector &v, double t) const
  {
    std::map<types::global_dof_index, double> bv;
    if (prm.use_mms)
      VectorTools::interpolate_boundary_values(
        dof_handler, 0, WaveExact::MMSExactSolution<dim>(prm.wave_speed, t), bv);
    else
      VectorTools::interpolate_boundary_values(
        dof_handler, 0, WaveExact::ExactSolution<dim>(prm.wave_speed, t), bv);

    TrilinosWrappers::MPI::Vector result(locally_owned_dofs, mpi_comm);
    for (const auto idx : locally_owned_dofs) {
      auto it = bv.find(idx);
      result[idx] = (it != bv.end()) ? it->second : v[idx];
    }
    result.compress(VectorOperation::insert);
    v = result;
  }

  template <int dim>
  void WaveEquationBase<dim>::add_mms_source(
    TrilinosWrappers::MPI::Vector &rhs, double t) const
  {
    if (!prm.use_mms) return;

    TrilinosWrappers::MPI::Vector f_nodal(locally_owned_dofs, mpi_comm);
    VectorTools::interpolate(dof_handler,
      WaveExact::MMSSource<dim>(prm.wave_speed, t), f_nodal);

    TrilinosWrappers::MPI::Vector Mf(locally_owned_dofs, mpi_comm);
    mass_matrix.vmult(Mf, f_nodal);

    // rhs += dt^2 * M * f   (Newmark RHS contribution, beta absorbed by caller
    rhs.add(1.0, Mf);
  }

  // ── Acceleration solve:  M*acc = -c^2*K*u (+ source)
  template <int dim>
  void WaveEquationBase<dim>::compute_acceleration(
    TrilinosWrappers::MPI::Vector &acc,
    const TrilinosWrappers::MPI::Vector &u) const
  {
    TrilinosWrappers::MPI::Vector rhs(locally_owned_dofs, mpi_comm);
    stiffness_matrix.vmult(rhs, u);
    rhs *= -(prm.wave_speed * prm.wave_speed);

    if (prm.use_mms) {
      TrilinosWrappers::MPI::Vector f_nodal(locally_owned_dofs, mpi_comm);
      VectorTools::interpolate(dof_handler,
        WaveExact::MMSSource<dim>(prm.wave_speed, time), f_nodal);
      TrilinosWrappers::MPI::Vector Mf(locally_owned_dofs, mpi_comm);
      mass_matrix.vmult(Mf, f_nodal);
      rhs.add(1.0, Mf);
    }

    acc = 0.;
    SolverControl sc(2000, 1e-12 * rhs.l2_norm() + 1e-14);
    TrilinosWrappers::SolverCG cg(sc);
    TrilinosWrappers::PreconditionJacobi prec;
    prec.initialize(mass_matrix);
    cg.solve(mass_matrix, acc, rhs, prec);
    constraints.distribute(acc);
  }

  // ── Discrete energy  E = ½(v^T M v + c^2 u^T K u)
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
    TimerOutput::Scope timer(timer_output, "05 error eval");
    TrilinosWrappers::MPI::Vector gu(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    gu = solution;
    Vector<double> cell_err(triangulation.n_active_cells());

    if (prm.use_mms)
      VectorTools::integrate_difference(
        dof_handler, gu, WaveExact::MMSExactSolution<dim>(prm.wave_speed, t),
        cell_err, QGauss<dim>(fe.degree + 2), VectorTools::L2_norm);
    else
      VectorTools::integrate_difference(
        dof_handler, gu, WaveExact::ExactSolution<dim>(prm.wave_speed, t),
        cell_err, QGauss<dim>(fe.degree + 2), VectorTools::L2_norm);

    return VectorTools::compute_global_error(
      triangulation, cell_err, VectorTools::L2_norm);
  }

  // ── H1 seminorm error  ||grad(u - u_h)||  
  // Physically: related to the potential-energy part of E_h.
  // Converges at O(h^p) — one order lower than L2's O(h^{p+1}).
  template <int dim>
  double WaveEquationBase<dim>::compute_h1_error(double t) const {
    TimerOutput::Scope timer(timer_output, "05 error eval");
    TrilinosWrappers::MPI::Vector gu(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    gu = solution;
    Vector<double> cell_err(triangulation.n_active_cells());

    if (prm.use_mms)
      VectorTools::integrate_difference(
        dof_handler, gu, WaveExact::MMSExactSolution<dim>(prm.wave_speed, t),
        cell_err, QGauss<dim>(fe.degree + 2), VectorTools::H1_seminorm);
    else
      VectorTools::integrate_difference(
        dof_handler, gu, WaveExact::ExactSolution<dim>(prm.wave_speed, t),
        cell_err, QGauss<dim>(fe.degree + 2), VectorTools::H1_seminorm);

    return VectorTools::compute_global_error(
      triangulation, cell_err, VectorTools::H1_seminorm);
  }

  // VTU output
  template <int dim>
  void WaveEquationBase<dim>::output_vtu(unsigned int step) const {
    TimerOutput::Scope timer(timer_output, "04 vtu output");

    TrilinosWrappers::MPI::Vector gu(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    TrilinosWrappers::MPI::Vector gv(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    gu = solution; gv = velocity;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(gu, "displacement");
    data_out.add_data_vector(gv, "velocity");

    TrilinosWrappers::MPI::Vector exact_owned(locally_owned_dofs, mpi_comm);
    if (prm.use_mms)
      VectorTools::interpolate(dof_handler,
        WaveExact::MMSExactSolution<dim>(prm.wave_speed, time), exact_owned);
    else
      VectorTools::interpolate(dof_handler,
        WaveExact::ExactSolution<dim>(prm.wave_speed, time), exact_owned);

    TrilinosWrappers::MPI::Vector exact_vec(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    exact_vec = exact_owned;
    data_out.add_data_vector(exact_vec, "exact_solution");

    // Pointwise error field 
    TrilinosWrappers::MPI::Vector err_owned(locally_owned_dofs, mpi_comm);
    err_owned = exact_owned;
    err_owned.add(-1.0, solution);
    TrilinosWrappers::MPI::Vector err_vec(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    err_vec = err_owned;
    data_out.add_data_vector(err_vec, "error");

    data_out.build_patches(fe.degree);
    data_out.write_vtu_with_pvtu_record(
      prm.output_dir + "/", "wave", step, mpi_comm, 4);
  }

  template <int dim>
  void WaveEquationBase<dim>::run() {
    if (prm.profiling)
      timer_output.reset();

    make_grid();
    setup_system();
    assemble_matrices();

    const double h   = 1.0 / std::pow(2.0, (double)prm.refinements);
    const double cfl = prm.wave_speed * dt * std::sqrt((double)dim) / h;
    pcout << "─────────────────────────────────────────\n"
          << " Scheme:    " << scheme_name() << "\n"
          << (prm.use_mms ? " Test case: MMS (manufactured solution)\n"
                          : " Test case: Standing wave\n")
          << " DoFs:      " << dof_handler.n_dofs() << "\n"
          << " h:         " << h    << "\n"
          << " dt:        " << dt   << "\n"
          << " CFL:       " << std::fixed << std::setprecision(3) << cfl << "\n"
          << "─────────────────────────────────────────\n";

    {
      TimerOutput::Scope t(timer_output, "01 mesh setup");
      if (prm.use_mms) {
        VectorTools::interpolate(dof_handler,
          WaveExact::MMSExactSolution<dim>(prm.wave_speed, 0.), solution);
        constraints.distribute(solution);
        VectorTools::interpolate(dof_handler,
          WaveExact::MMSInitialVelocity<dim>(), velocity);
        constraints.distribute(velocity);
      } else {
        VectorTools::interpolate(dof_handler,
          WaveExact::ExactSolution<dim>(prm.wave_speed, 0.), solution);
        constraints.distribute(solution);
        VectorTools::interpolate(dof_handler,
          WaveExact::InitialVelocity<dim>(), velocity);
        constraints.distribute(velocity);
      }
    }

    compute_acceleration(acceleration, solution);
    init_scheme_state();

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    if (rank == 0) {
      std::filesystem::create_directories(prm.output_dir);
      energy_log.open(prm.output_dir + "/energy.csv");
      energy_log << "step,time,energy,energy_ratio\n";
      error_log.open(prm.output_dir + "/error.csv");
      error_log << "step,time,l2_error,h1_error\n";
    }

    const double E0 = compute_energy();
    pcout << std::fixed << std::setprecision(4)
          << "t = " << time
          << "  E = " << E0
          << "  L2 = " << compute_l2_error(0.)
          << "  H1 = " << compute_h1_error(0.) << "\n";
    if (prm.output_every > 0) output_vtu(0);

    Timer timer;
    double total_time = 0.;
    const unsigned int n_steps =
      static_cast<unsigned int>(std::round(prm.final_time / dt));
    const unsigned int print_every = std::max(1u, n_steps / 5);
    const unsigned int log_every   = std::max(1u, n_steps / 20);

    for (step_number = 1; step_number <= n_steps; ++step_number) {
      time = step_number * dt;
      timer.restart();
      {
        TimerOutput::Scope t(timer_output, "03 time integration");
        advance_one_step();
      }
      total_time += timer.wall_time();

      const double E = compute_energy();

      if (step_number % print_every == 0)
        pcout << "t = " << std::fixed    << std::setprecision(4) << time
              << "  E/E0 = " << std::scientific << E / E0
              << "  L2 = "   << compute_l2_error(time)
              << "  H1 = "   << compute_h1_error(time) << "\n";

      if (step_number % log_every == 0) {
        const double l2 = compute_l2_error(time);
        const double h1 = compute_h1_error(time);
        if (rank == 0)
          error_log << step_number << "," << time << "," << l2 << "," << h1 << "\n";
      }
      if (rank == 0)
        energy_log << step_number << "," << time << "," << E << "," << E/E0 << "\n";

      if (prm.output_every > 0 && step_number % prm.output_every == 0)
        output_vtu(step_number);
    }

    const double final_l2 = compute_l2_error(prm.final_time);
    const double final_h1 = compute_h1_error(prm.final_time);
    const double final_er = compute_energy() / E0;

    pcout << "\nFinal L2 error: " << std::scientific << std::setprecision(6) << final_l2 << "\n"
          << "Final H1 error: " << std::scientific << std::setprecision(6) << final_h1 << "\n"
          << "Final E/E0: "     << std::scientific << std::setprecision(6) << final_er << "\n"
          << "Wall time: "      << std::fixed      << std::setprecision(3) << total_time << "\n";

    if (rank == 0) {
      std::ofstream cv(prm.output_dir + "/convergence.csv", std::ios::app);
      cv.seekp(0, std::ios::end);
      if (cv.tellp() == 0)
        cv << "scheme,fe_degree,refinements,h,dt,dofs,final_l2_error,final_h1_error,"
              "final_energy_ratio,wall_time,use_mms\n";
      cv << scheme_name() << ","
         << prm.fe_degree << ","
         << prm.refinements << ","
         << h << ","
         << dt << ","
         << dof_handler.n_dofs() << ","
         << final_l2 << ","
         << final_h1 << ","
         << final_er << ","
         << total_time << ","
         << (prm.use_mms ? "1" : "0") << "\n";
    }

    energy_log.close();
    error_log.close();

    if (prm.profiling)
      timer_output.print_summary();
  }

} 
