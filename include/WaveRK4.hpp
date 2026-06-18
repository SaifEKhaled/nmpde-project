#pragma once
/* ============================================================
 * WaveRK4.hpp
 *
 * Classical 4th-order Runge–Kutta solver for the wave equation.
 *
 * The 2nd-order PDE is written as the 1st-order system:
 *   d/dt [u]   [       v        ]
 *        [v] = [M^{-1}(-c^2 K u)]
 *
 * and the standard RK4 tableau is applied:
 *   k1 = f(y^n)
 *   k2 = f(y^n + dt/2 * k1)
 *   k3 = f(y^n + dt/2 * k2)
 *   k4 = f(y^n + dt   * k3)
 *   y^{n+1} = y^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
 *
 * Properties:
 *   - 4th-order in time — best accuracy per step
 *   - Explicit — 4 mass-matrix solves per step
 *   - Not symplectic — but energy drift is O(dt^4) * T, very small
 *   - Conditionally stable (stability region covers imaginary axis
 *     up to |lambda*dt| ~ 2.83, typically safe within CFL)
 * ============================================================ */

#include "WaveEquationBase.hpp"

namespace WaveSolver {
  using namespace dealii;

  template <int dim>
  class WaveRK4 : public WaveEquationBase<dim> {
  public:
    WaveRK4(const Parameters &prm);

  protected:
    void advance_one_step() override;
    std::string scheme_name() const override { return "RK4 (4th-order Runge-Kutta)"; }

  private:
    using Base = WaveEquationBase<dim>;
  };

  // ── Constructor ────────────────────────────────────────────
  template <int dim>
  WaveRK4<dim>::WaveRK4(const Parameters &prm)
    : WaveEquationBase<dim>(prm)
  {}

  // ── Single time step ───────────────────────────────────────
  template <int dim>
  void WaveRK4<dim>::advance_one_step() {
    const double dt_ = Base::dt;
    auto alloc = [&]() {
      return TrilinosWrappers::MPI::Vector(Base::locally_owned_dofs, Base::mpi_comm);
    };

    // Stage 1
    auto ku1 = alloc(); auto kv1 = alloc();
    ku1 = Base::velocity;
    Base::compute_acceleration(kv1, Base::solution);

    // Stage 2
    auto u2 = alloc(); auto v2 = alloc();
    u2 = Base::solution; u2.add(dt_/2., ku1);
    v2 = Base::velocity; v2.add(dt_/2., kv1);
    auto ku2 = alloc(); auto kv2 = alloc();
    ku2 = v2;
    Base::compute_acceleration(kv2, u2);

    // Stage 3
    auto u3 = alloc(); auto v3 = alloc();
    u3 = Base::solution; u3.add(dt_/2., ku2);
    v3 = Base::velocity; v3.add(dt_/2., kv2);
    auto ku3 = alloc(); auto kv3 = alloc();
    ku3 = v3;
    Base::compute_acceleration(kv3, u3);

    // Stage 4
    auto u4 = alloc(); auto v4 = alloc();
    u4 = Base::solution; u4.add(dt_, ku3);
    v4 = Base::velocity; v4.add(dt_, kv3);
    auto ku4 = alloc(); auto kv4 = alloc();
    ku4 = v4;
    Base::compute_acceleration(kv4, u4);

    // Combine
    Base::solution.add(dt_/6., ku1); Base::solution.add(dt_/3., ku2);
    Base::solution.add(dt_/3., ku3); Base::solution.add(dt_/6., ku4);
    Base::velocity.add(dt_/6., kv1); Base::velocity.add(dt_/3., kv2);
    Base::velocity.add(dt_/3., kv3); Base::velocity.add(dt_/6., kv4);

    Base::apply_dirichlet_bc(Base::solution, Base::time + dt_);
    Base::apply_dirichlet_bc(Base::velocity, Base::time + dt_);

    // Update stored acceleration for energy logging
    Base::compute_acceleration(Base::acceleration, Base::solution);
  }

} // namespace WaveSolver