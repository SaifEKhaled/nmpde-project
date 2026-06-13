#pragma once
/* ============================================================
 * WaveExact.hpp
 *
 * Exact solution for the 2-D standing wave test case:
 *
 *   u(x,y,t) = cos(omega*t) * sin(pi*x) * sin(pi*y)
 *   omega     = c * pi * sqrt(2)
 *
 * Satisfies:
 *   u_tt - c^2 * Delta u = 0   on (0,1)^2 x (0,T]
 *   u = 0                       on boundary (homogeneous Dirichlet)
 *   u(0)   = sin(pi*x)*sin(pi*y)
 *   u_t(0) = 0
 * ============================================================ */

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <cmath>

namespace WaveExact {
  using namespace dealii;

  inline double omega(double c) {
    return c * numbers::PI * std::sqrt(2.0);
  }

  template <int dim>
  class ExactSolution : public Function<dim> {
  public:
    ExactSolution(double c, double t = 0.)
      : Function<dim>(1, t), c_(c) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double t = this->get_time();
      double spatial = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        spatial *= std::sin(numbers::PI * p[d]);
      return std::cos(omega(c_) * t) * spatial;
    }

  private:
    double c_;
  };

  // u_t(x,y,0) = 0  (cosine in time, zero derivative at t=0)
  template <int dim>
  class InitialVelocity : public Function<dim> {
  public:
    InitialVelocity() : Function<dim>(1, 0.) {}
    double value(const Point<dim> &, const unsigned int = 0) const override {
      return 0.0;
    }
  };

  // du/dt = -omega * sin(omega*t) * sin(pi*x) * sin(pi*y)
  // Used to set consistent velocity BCs in WaveTheta.
  template <int dim>
  class ExactVelocity : public Function<dim> {
  public:
    ExactVelocity(double c, double t = 0.)
      : Function<dim>(1, t), c_(c) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double t = this->get_time();
      double spatial = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        spatial *= std::sin(numbers::PI * p[d]);
      return -omega(c_) * std::sin(omega(c_) * t) * spatial;
    }

  private:
    double c_;
  };

}