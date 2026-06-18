#pragma once
/* ============================================================
 * WaveLeapfrog.hpp
 *
 * Störmer–Verlet (Leapfrog) solver for the wave equation.
 *
 * Algorithm (position-Verlet form):
 *   a^n     = M^{-1}(-c^2 K u^n)
 *   u^{n+1} = 2*u^n - u^{n-1} + dt^2 * a^n
 *   v^n     = (u^{n+1} - u^{n-1}) / (2*dt)    [centred velocity]
 *
 * Bootstrap: u^{-1} = u^0 - dt*v^0 + (dt^2/2)*a^0
 *            (since v^0 = 0 here, simplifies to u^0 + (dt^2/2)*a^0)
 *
 * Properties:
 *   - 2nd-order in time
 *   - Explicit — no linear solve per step (just one M solve for a^n)
 *   - Symplectic — conserves a modified energy; E/E0 oscillates near 1
 *   - Conditionally stable: dt <= h / (c * sqrt(dim))
 *   - Roughly 4× faster than implicit methods per step
 * ============================================================ */

#include "WaveEquationBase.hpp"

namespace WaveSolver {
  using namespace dealii;

  template <int dim>
  class WaveLeapfrog : public WaveEquationBase<dim> {
  public:
    WaveLeapfrog(const Parameters &prm);

  protected:
    void advance_one_step() override;
    void init_scheme_state() override;
    std::string scheme_name() const override { return "Leapfrog (Störmer-Verlet)"; }

  private:
    TrilinosWrappers::MPI::Vector old_solution_;  // u^{n-1}

    using Base = WaveEquationBase<dim>;
  };

  // ── Constructor ────────────────────────────────────────────
  template <int dim>
  WaveLeapfrog<dim>::WaveLeapfrog(const Parameters &prm)
    : WaveEquationBase<dim>(prm)
  {}

  // ── Bootstrap u^{-1} ───────────────────────────────────────
  template <int dim>
  void WaveLeapfrog<dim>::init_scheme_state() {
    old_solution_.reinit(Base::locally_owned_dofs, Base::mpi_comm);
    // u^{-1} = u^0 + (dt^2/2)*a^0   (v^0 = 0)
    old_solution_ = Base::solution;
    old_solution_.add(-(Base::dt * Base::dt / 2.0), Base::acceleration);
  }

  // ── Single time step ───────────────────────────────────────
  template <int dim>
  void WaveLeapfrog<dim>::advance_one_step() {
    const double dt_ = Base::dt;

    // a^n = M^{-1}(-c^2 K u^n)
    TrilinosWrappers::MPI::Vector acc(Base::locally_owned_dofs, Base::mpi_comm);
    Base::compute_acceleration(acc, Base::solution);

    // u^{n+1} = 2*u^n - u^{n-1} + dt^2 * a^n
    TrilinosWrappers::MPI::Vector new_u(Base::locally_owned_dofs, Base::mpi_comm);
    new_u = Base::solution;
    new_u *= 2.0;
    new_u.add(-1.0, old_solution_);
    new_u.add(dt_ * dt_, acc);
    Base::apply_dirichlet_bc(new_u, Base::time + dt_);

    // Centred velocity v^n = (u^{n+1} - u^{n-1}) / (2*dt)
    Base::velocity = new_u;
    Base::velocity.add(-1.0, old_solution_);
    Base::velocity *= 1.0 / (2.0 * dt_);

    old_solution_      = Base::solution;
    Base::solution     = new_u;
    Base::acceleration = acc;
  }

} // namespace WaveSolver