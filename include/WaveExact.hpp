#pragma once
/* ============================================================
 * WaveExact.hpp
 *
 * Two test cases:
 *
 * 1. STANDING WAVE (default)
 *    u(x,y,t) = cos(omega*t) * sin(pi*x) * sin(pi*y)
 *    omega = c*pi*sqrt(2)
 *    Homogeneous source f = 0.
 *
 * 2. MANUFACTURED SOLUTION (MMS)
 *    u_mms(x,y,t) = sin(pi*x) * sin(pi*y) * sin(pi*t)
 *    This does NOT satisfy the homogeneous wave equation,
 *    so we add a source term:
 *      f = u_tt - c^2*Delta(u)
 *        = -pi^2*sin(pi*x)*sin(pi*y)*sin(pi*t)
 *          + 2*c^2*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*t)
 *        = pi^2*(2*c^2 - 1)*sin(pi*x)*sin(pi*y)*sin(pi*t)
 *    with BCs u_mms = 0 on boundary (sin(pi*0)=sin(pi*1)=0).
 *    IC: u(0) = 0, v(0) = pi*sin(pi*x)*sin(pi*y)
 *
 * The MMS test is more rigorous: it verifies convergence
 * independent of any accidental cancellation from the
 * standing-wave symmetry.
 * ============================================================ */

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <cmath>

namespace WaveExact {
  using namespace dealii;

  // ── Helpers ───────────────────────────────────────────────
  inline double omega(double c) {
    return c * numbers::PI * std::sqrt(2.0);
  }

  // ── Standing wave ─────────────────────────────────────────
  template <int dim>
  class ExactSolution : public Function<dim> {
  public:
    ExactSolution(double c, double t = 0.)
      : Function<dim>(1, t), c_(c) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double t = this->get_time();
      double s = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        s *= std::sin(numbers::PI * p[d]);
      return std::cos(omega(c_) * t) * s;
    }

    // grad u = cos(omega*t) * [pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
  const double t = this->get_time();
  const double ct = std::cos(omega(c_) * t);

  Tensor<1, dim> g;

  for (unsigned int d = 0; d < dim; ++d) {
    double term = numbers::PI;

    for (unsigned int e = 0; e < dim; ++e)
      term *= (e == d)
                ? std::cos(numbers::PI * p[e])
                : std::sin(numbers::PI * p[e]);

    g[d] = ct * term;
  }

  return g;
}

  private:
    double c_;
  };

  template <int dim>
  class InitialVelocity : public Function<dim> {
  public:
    InitialVelocity() : Function<dim>(1, 0.) {}
    double value(const Point<dim> &, const unsigned int = 0) const override {
      return 0.0;
    }
  };

  template <int dim>
  class ExactVelocity : public Function<dim> {
  public:
    ExactVelocity(double c, double t = 0.)
      : Function<dim>(1, t), c_(c) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double t = this->get_time();
      double s = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        s *= std::sin(numbers::PI * p[d]);
      return -omega(c_) * std::sin(omega(c_) * t) * s;
    }

  private:
    double c_;
  };

  // ── Manufactured solution ─────────────────────────────────
  // u_mms = sin(pi*x)*sin(pi*y)*sin(pi*t)
  template <int dim>
  class MMSExactSolution : public Function<dim> {
  public:
    MMSExactSolution(double /*c*/ = 1., double t = 0.)
      : Function<dim>(1, t) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double t = this->get_time();
      double s = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        s *= std::sin(numbers::PI * p[d]);
      return s * std::sin(numbers::PI * t);
    }

    // grad u = sin(pi*t) * [pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
  const double t = this->get_time();
  const double st = std::sin(numbers::PI * t);

  Tensor<1, dim> g;

  for (unsigned int d = 0; d < dim; ++d) {
    double term = numbers::PI;

    for (unsigned int e = 0; e < dim; ++e)
      term *= (e == d)
                ? std::cos(numbers::PI * p[e])
                : std::sin(numbers::PI * p[e]);

    g[d] = st * term;
  }

  return g;
}
  };

  // v_mms = pi*sin(pi*x)*sin(pi*y)*cos(pi*t)
  template <int dim>
  class MMSInitialVelocity : public Function<dim> {
  public:
    MMSInitialVelocity() : Function<dim>(1, 0.) {}
    double value(const Point<dim> &p, const unsigned int = 0) const override {
      double s = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        s *= std::sin(numbers::PI * p[d]);
      return numbers::PI * s;   // cos(pi*0) = 1
    }
  };

  // Source term:  f = pi^2*(2*c^2 - 1)*sin(pi*x)*sin(pi*y)*sin(pi*t)
  template <int dim>
  class MMSSource : public Function<dim> {
  public:
    MMSSource(double c, double t = 0.)
      : Function<dim>(1, t), c_(c) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double t = this->get_time();
      double s = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        s *= std::sin(numbers::PI * p[d]);
      // u_tt = -pi^2 * s * sin(pi*t)
      // -c^2*Delta(u) = +2*c^2*pi^2 * s * sin(pi*t)  (in 2D)
      const double coeff = numbers::PI * numbers::PI
                         * (2.0 * c_ * c_ - 1.0);
      return coeff * s * std::sin(numbers::PI * t);
    }

  private:
    double c_;
  };

} // namespace WaveExact