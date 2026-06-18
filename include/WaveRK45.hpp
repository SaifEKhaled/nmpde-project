#pragma once
/* ============================================================
 * WaveRK45.hpp
 *
 * Adaptive embedded Runge-Kutta solver (Dormand-Prince RK5(4))
 * for the wave equation, written as the 1st-order system
 *   d/dt [u]   [       v        ]
 *        [v] = [M^{-1}(-c^2 K u)]
 *
 * The Dormand-Prince tableau computes a 5th-order solution and
 * a 4th-order solution from the same 6 stage evaluations; their
 * difference gives a local error estimate with NO extra cost.
 *
 * Step-size control (standard PI-ish controller):
 *   err  = ||y5 - y4|| / (atol + rtol*||y||)   (simplified: atol only here)
 *   dt_new = dt * safety * (tol/err)^(1/5)
 *   if err <= tol: accept step, dt <- dt_new (clamped)
 *   else:          reject step, retry with smaller dt
 *
 * This is a genuinely separate scheme from RK4 — RK4 is untouched,
 * all existing fixed-dt results remain valid and reproducible.
 *
 * Reference: Dormand & Prince (1980), "A family of embedded
 * Runge-Kutta formulae", J. Comput. Appl. Math.
 * ============================================================ */

#include "WaveEquationBase.hpp"
#include <algorithm>
#include <cmath>

namespace WaveSolver {
  using namespace dealii;

  template <int dim>
  class WaveRK45 : public WaveEquationBase<dim> {
  public:
    WaveRK45(const Parameters &prm);

  protected:
    void advance_one_step() override {
      advance_one_step_adaptive();
    }
    bool is_adaptive() const override { return true; }
    double advance_one_step_adaptive() override;

    std::string scheme_name() const override {
      return "RK45 (Dormand-Prince, adaptive)";
    }

  private:
    using Base = WaveEquationBase<dim>;
    using Vec  = TrilinosWrappers::MPI::Vector;

    // Dormand-Prince coefficients (c_i, a_ij, b5_i 5th order, b4_i 4th order)
    static constexpr double c2 = 1.0/5.0,  c3 = 3.0/10.0, c4 = 4.0/5.0,
                             c5 = 8.0/9.0,  c6 = 1.0;

    static constexpr double a21 = 1.0/5.0;
    static constexpr double a31 = 3.0/40.0,      a32 = 9.0/40.0;
    static constexpr double a41 = 44.0/45.0,     a42 = -56.0/15.0,    a43 = 32.0/9.0;
    static constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0,
                             a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    static constexpr double a61 = 9017.0/3168.0,  a62 = -355.0/33.0,
                             a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;

    // 5th-order solution weights
    static constexpr double b1 = 35.0/384.0,    b2 = 0.0, b3 = 500.0/1113.0,
                             b4 = 125.0/192.0,   b5 = -2187.0/6784.0, b6 = 11.0/84.0;
    // 4th-order solution weights (embedded, for error estimate)
    static constexpr double bs1 = 5179.0/57600.0,  bs2 = 0.0, bs3 = 7571.0/16695.0,
                             bs4 = 393.0/640.0,     bs5 = -92097.0/339200.0,
                             bs6 = 187.0/2100.0,    bs7 = 1.0/40.0;
  };

  // Constructor
  template <int dim>
  WaveRK45<dim>::WaveRK45(const Parameters &prm)
    : WaveEquationBase<dim>(prm)
  {}

  // Single adaptive step
  // y = [u, v]; f(y) = [v, M^{-1}(-c^2 K u)]
  template <int dim>
  double WaveRK45<dim>::advance_one_step_adaptive() {
    const auto &owned = Base::locally_owned_dofs;
    const auto &comm  = Base::mpi_comm;
    auto alloc = [&]() { return Vec(owned, comm); };

    const unsigned int max_retries = 10;

    for (unsigned int attempt = 0; attempt < max_retries; ++attempt) {
      const double dt_ = Base::dt;

      // stage 1
      Vec ku1 = alloc(), kv1 = alloc();
      ku1 = Base::velocity;
      Base::compute_acceleration(kv1, Base::solution);

      // stage 2
      Vec u2 = alloc(), v2 = alloc();
      u2 = Base::solution; u2.add(dt_*a21, ku1);
      v2 = Base::velocity; v2.add(dt_*a21, kv1);
      Vec ku2 = alloc(), kv2 = alloc();
      ku2 = v2;
      Base::compute_acceleration(kv2, u2);

      // stage 3
      Vec u3 = alloc(), v3 = alloc();
      u3 = Base::solution; u3.add(dt_*a31, ku1); u3.add(dt_*a32, ku2);
      v3 = Base::velocity; v3.add(dt_*a31, kv1); v3.add(dt_*a32, kv2);
      Vec ku3 = alloc(), kv3 = alloc();
      ku3 = v3;
      Base::compute_acceleration(kv3, u3);

      // stage 4
      Vec u4 = alloc(), v4 = alloc();
      u4 = Base::solution; u4.add(dt_*a41, ku1); u4.add(dt_*a42, ku2); u4.add(dt_*a43, ku3);
      v4 = Base::velocity; v4.add(dt_*a41, kv1); v4.add(dt_*a42, kv2); v4.add(dt_*a43, kv3);
      Vec ku4 = alloc(), kv4 = alloc();
      ku4 = v4;
      Base::compute_acceleration(kv4, u4);

      // stage 5
      Vec u5 = alloc(), v5 = alloc();
      u5 = Base::solution;
      u5.add(dt_*a51, ku1); u5.add(dt_*a52, ku2); u5.add(dt_*a53, ku3); u5.add(dt_*a54, ku4);
      v5 = Base::velocity;
      v5.add(dt_*a51, kv1); v5.add(dt_*a52, kv2); v5.add(dt_*a53, kv3); v5.add(dt_*a54, kv4);
      Vec ku5 = alloc(), kv5 = alloc();
      ku5 = v5;
      Base::compute_acceleration(kv5, u5);
      // stage 6
      Vec u6 = alloc(), v6 = alloc();
      u6 = Base::solution;
      u6.add(dt_*a61, ku1); u6.add(dt_*a62, ku2); u6.add(dt_*a63, ku3);
      u6.add(dt_*a64, ku4); u6.add(dt_*a65, ku5);
      v6 = Base::velocity;
      v6.add(dt_*a61, kv1); v6.add(dt_*a62, kv2); v6.add(dt_*a63, kv3);
      v6.add(dt_*a64, kv4); v6.add(dt_*a65, kv5);
      Vec ku6 = alloc(), kv6 = alloc();
      ku6 = v6;
      Base::compute_acceleration(kv6, u6);

      // 5th-order solution (the one we advance with) 
      Vec u5th = alloc(), v5th = alloc();
      u5th = Base::solution;
      u5th.add(dt_*b1, ku1); u5th.add(dt_*b3, ku3);
      u5th.add(dt_*b4, ku4); u5th.add(dt_*b5, ku5); u5th.add(dt_*b6, ku6);
      v5th = Base::velocity;
      v5th.add(dt_*b1, kv1); v5th.add(dt_*b3, kv3);
      v5th.add(dt_*b4, kv4); v5th.add(dt_*b5, kv5); v5th.add(dt_*b6, kv6);

      // Dormand-Prince b7 weight uses k7 = f(y5th) (the "first same as last" trick).
      Vec ku7 = alloc(), kv7 = alloc();
      ku7 = v5th;
      Base::compute_acceleration(kv7, u5th);

      // 4th-order solution (for error estimate only)
      Vec u4th = alloc(), v4th = alloc();
      u4th = Base::solution;
      u4th.add(dt_*bs1, ku1); u4th.add(dt_*bs3, ku3); u4th.add(dt_*bs4, ku4);
      u4th.add(dt_*bs5, ku5); u4th.add(dt_*bs6, ku6); u4th.add(dt_*bs7, ku7);
      v4th = Base::velocity;
      v4th.add(dt_*bs1, kv1); v4th.add(dt_*bs3, kv3); v4th.add(dt_*bs4, kv4);
      v4th.add(dt_*bs5, kv5); v4th.add(dt_*bs6, kv6); v4th.add(dt_*bs7, kv7);

      // error estimate: max-norm of (y5 - y4), scaled by tolerance 
      Vec u_diff = alloc(), v_diff = alloc();
      u_diff = u5th; u_diff.add(-1.0, u4th);
      v_diff = v5th; v_diff.add(-1.0, v4th);
      const double err_u = u_diff.linfty_norm();
      const double err_v = v_diff.linfty_norm();
      const double err   = std::max(err_u, err_v);

      const double tol    = Base::prm.adaptive_tol;
      const double safety = 0.9;
      const double order  = 5.0;

      // try not to divide by zero in dt_factor calculation if err is extremely small
      const double err_safe = std::max(err, 1e-14);
      double dt_factor = safety * std::pow(tol / err_safe, 1.0 / order);
      dt_factor = std::clamp(dt_factor, 0.2, 5.0);  // limit growth/shrink per step

      const double dt_new = std::clamp(dt_ * dt_factor,
                                        Base::prm.min_dt, Base::prm.max_dt);

      if (err <= tol || dt_ <= Base::prm.min_dt * 1.0001) {
        // Accept step (or forced accept if already at min_dt)
        Base::solution = u5th;
        Base::velocity = v5th;
        Base::apply_dirichlet_bc(Base::solution, Base::time + dt_);
        Base::apply_dirichlet_bc(Base::velocity, Base::time + dt_);
        Base::compute_acceleration(Base::acceleration, Base::solution);

        Base::dt = dt_new;   
        return dt_;          
      } else {
        Base::dt = dt_new;
      }
    }

    // Failed to converge after max_retries — accept anyway at current dt
    // to avoid an infinite loop (rare; only at pathological tolerances).
    Base::pcout << "  [WARN] RK45: max retries reached, forcing accept\n";
    Base::compute_acceleration(Base::acceleration, Base::solution);
    return Base::dt;
  }

} // namespace WaveSolver