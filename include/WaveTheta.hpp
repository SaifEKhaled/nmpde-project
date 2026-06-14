#pragma once
/* ============================================================
 * WaveTheta.hpp
 *
 * Newmark-beta time integrator for the wave equation.
 *
 * Formulation (displacement only — one linear solve per step):
 *
 *   Predict:
 *     u*       = u^n + dt*v^n + dt^2*(0.5-beta)*a^n
 *     v*       = v^n + dt*(1-gamma)*a^n
 *
 *   Solve:
 *     [M + beta*dt^2*c^2*K] u^{n+1} = M*u* - beta*dt^2*c^2*K*(u*-u*)
 *     i.e.  (M + beta*dt^2*c^2*K) u^{n+1} = M*u*
 *           actually:  rhs = M*u_pred - ... let me write it cleanly:
 *
 *   [M + beta*dt^2*c^2*K] u^{n+1} = M*u^n + dt*M*v^n
 *                                   + dt^2*(0.5-beta)*M*a^n
 *
 *   Correct:
 *     a^{n+1} = -c^2 * M^{-1} * K * u^{n+1}
 *     v^{n+1} = v* + dt*gamma*a^{n+1}
 *
 * Parameter mapping:
 *   beta=0.25, gamma=0.5  -> Average acceleration (CN-equivalent, energy-conserving)
 *   beta=0.0,  gamma=0.5  -> Central difference / Leapfrog (explicit, CFL-limited)
 *   beta=0.25, gamma=1.0  -> Not standard, more dissipative
 *
 * For the theta parameter in the .prm file:
 *   theta -> beta = theta/2,  gamma = 0.5  (so theta=0.5 -> beta=0.25, energy-conserving)
 *   theta=1.0 -> beta=0.5, gamma=0.5  (implicit, dissipative, 1st order)
 *   theta=0.0 -> beta=0.0, gamma=0.5  (explicit, CFL-limited)
 *
 * This formulation conserves the discrete energy
 *   E_h = 0.5 * (v^T M v + c^2 u^T K u)
 * exactly for gamma=0.5, beta=0.25 (up to solver tolerance).
 * ============================================================ */

#include "WaveEquationBase.hpp"

namespace WaveSolver {
  using namespace dealii;

  template <int dim>
  class WaveTheta : public WaveEquationBase<dim> {
  public:
    WaveTheta(const Parameters &prm);

  protected:
    void advance_one_step() override;
    void init_scheme_state() override;
    std::string scheme_name() const override;

  private:
    // Newmark parameters derived from prm.theta:
    //   beta  = theta / 2
    //   gamma = 0.5
    double beta_;
    double gamma_;

    // Pre-assembled:  A = M + beta*dt^2*c^2*K
    TrilinosWrappers::SparseMatrix system_matrix_;

    using Base = WaveEquationBase<dim>;
  };

  // ── Constructor ────────────────────────────────────────────
  template <int dim>
  WaveTheta<dim>::WaveTheta(const Parameters &prm)
    : WaveEquationBase<dim>(prm),
      beta_ (prm.theta / 2.0),
      gamma_(0.5)
  {}

  // ── Build system matrix once ───────────────────────────────
  template <int dim>
  void WaveTheta<dim>::init_scheme_state() {
    system_matrix_.copy_from(Base::mass_matrix);
    const double alpha = beta_
                       * Base::dt * Base::dt
                       * Base::prm.wave_speed * Base::prm.wave_speed;
    system_matrix_.add(alpha, Base::stiffness_matrix);
    system_matrix_.compress(VectorOperation::add);
  }

  // ── Scheme label ───────────────────────────────────────────
  template <int dim>
  std::string WaveTheta<dim>::scheme_name() const {
    const double th = Base::prm.theta;
    if (th == 0.5) return "Crank-Nicolson / Newmark (beta=0.25, energy-conserving)";
    if (th == 1.0) return "Newmark (beta=0.5, dissipative)";
    if (th == 0.0) return "Newmark (beta=0, explicit / central difference)";
    return "Newmark (beta=" + std::to_string(beta_) + ")";
  }

  // ── Single time step ───────────────────────────────────────
  template <int dim>
  void WaveTheta<dim>::advance_one_step() {
    // const double c2  = Base::prm.wave_speed * Base::prm.wave_speed;
    const double dt_ = Base::dt;

    // ── Predictor ─────────────────────────────────────────
    // u_pred = u^n + dt*v^n + dt^2*(0.5-beta)*a^n
    // v_pred = v^n + dt*(1-gamma)*a^n
    TrilinosWrappers::MPI::Vector u_pred(Base::locally_owned_dofs, Base::mpi_comm);
    TrilinosWrappers::MPI::Vector v_pred(Base::locally_owned_dofs, Base::mpi_comm);

    u_pred = Base::solution;
    u_pred.add(dt_,                       Base::velocity);
    u_pred.add(dt_*dt_*(0.5 - beta_),     Base::acceleration);

    v_pred = Base::velocity;
    v_pred.add(dt_*(1.0 - gamma_),        Base::acceleration);

    // ── RHS:  M * u_pred ──────────────────────────────────
    TrilinosWrappers::MPI::Vector rhs(Base::locally_owned_dofs, Base::mpi_comm);
    Base::mass_matrix.vmult(rhs, u_pred);

    if (Base::prm.use_mms) {
      // Newmark RHS source contribution: dt^2 * beta * M * f(t^{n+1})
      TrilinosWrappers::MPI::Vector f_nodal(Base::locally_owned_dofs, Base::mpi_comm);
      VectorTools::interpolate(Base::dof_handler,
        WaveExact::MMSSource<dim>(Base::prm.wave_speed, Base::time + dt_), f_nodal);

      TrilinosWrappers::MPI::Vector Mf(Base::locally_owned_dofs, Base::mpi_comm);
      Base::mass_matrix.vmult(Mf, f_nodal);

      rhs.add(dt_ * dt_ * beta_, Mf);
    }

    // ── Apply Dirichlet BCs to system and RHS ─────────────
    std::map<types::global_dof_index, double> bv;
    if (Base::prm.use_mms)
      VectorTools::interpolate_boundary_values(
        Base::dof_handler, 0,
        WaveExact::MMSExactSolution<dim>(
          Base::prm.wave_speed,
          Base::time + dt_), bv);
    else
      VectorTools::interpolate_boundary_values(
        Base::dof_handler, 0,
        WaveExact::ExactSolution<dim>(
          Base::prm.wave_speed,
          Base::time + dt_), bv);

    TrilinosWrappers::SparseMatrix sys_copy;
    sys_copy.copy_from(system_matrix_);
    TrilinosWrappers::MPI::Vector new_u(Base::locally_owned_dofs, Base::mpi_comm);
    MatrixTools::apply_boundary_values(bv, sys_copy, new_u, rhs, false);

    // ── Solve  (M + beta*dt^2*c^2*K) u^{n+1} = rhs ───────
    SolverControl sc(5000, 1e-12 * rhs.l2_norm() + 1e-14);
    TrilinosWrappers::SolverCG cg(sc);
    TrilinosWrappers::PreconditionAMG prec;
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    prec.initialize(sys_copy, amg_data);
    cg.solve(sys_copy, new_u, rhs, prec);
    Base::constraints.distribute(new_u);

    // ── Corrector ─────────────────────────────────────────
    // a^{n+1} = M^{-1}(-c^2 K u^{n+1})
    // v^{n+1} = v_pred + dt*gamma*a^{n+1}
    TrilinosWrappers::MPI::Vector acc_new(Base::locally_owned_dofs, Base::mpi_comm);
    Base::compute_acceleration(acc_new, new_u);

    TrilinosWrappers::MPI::Vector new_v(Base::locally_owned_dofs, Base::mpi_comm);
    new_v = v_pred;
    new_v.add(dt_ * gamma_, acc_new);
    Base::constraints.distribute(new_v);

    // ── Advance ───────────────────────────────────────────
    Base::solution     = new_u;
    Base::velocity     = new_v;
    Base::acceleration = acc_new;
  }

} // namespace WaveSolver