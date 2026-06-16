/* ============================================================
 * main.cc
 *
 * The "Scheme" field in the .prm file selects the time integrator:
 *   CN       -> Crank-Nicolson (theta=0.5, 2nd order, implicit)
 *   BE       -> Backward Euler (theta=1.0, 1st order, dissipative)
 *   FE       -> Forward Euler  (theta=0.0, 1st order, explicit)
 *   Leapfrog -> Störmer-Verlet (2nd order, symplectic, explicit)
 *   RK4      -> Runge-Kutta 4  (4th order, explicit)
 *
 * The "Dimension" field (2 or 3) selects the spatial dimension.
 * All exact solutions, source terms, and the CFL formula are
 * dimension-generic (standing wave = product of sines over `dim`
 * coordinates, omega = c*pi*sqrt(dim)).
 * ============================================================ */

#include "Parameters.hpp"
#include "WaveTheta.hpp"
#include "WaveLeapfrog.hpp"
#include "WaveRK4.hpp"
#include "WaveRK45.hpp"

#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <memory>

template <int dim>
void run_solver(const Parameters &prm) {
  std::unique_ptr<WaveSolver::WaveEquationBase<dim>> solver;

  if (prm.scheme == "Theta") {
    solver = std::make_unique<WaveSolver::WaveTheta<dim>>(prm);
  } else if (prm.scheme == "Leapfrog") {
    solver = std::make_unique<WaveSolver::WaveLeapfrog<dim>>(prm);
  } else if (prm.scheme == "RK4") {
    solver = std::make_unique<WaveSolver::WaveRK4<dim>>(prm);
  } else if (prm.scheme == "RK45") {
    solver = std::make_unique<WaveSolver::WaveRK45<dim>>(prm);
  } else {
    throw std::runtime_error("Unknown scheme: " + prm.scheme);
  }

  solver->run();
}

int main(int argc, char **argv) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(
    argc, argv, dealii::numbers::invalid_unsigned_int);
  dealii::deallog.depth_console(0);

  try {
    Parameters prm;
    dealii::ParameterHandler handler;
    prm.declare(handler);

    const std::string prm_file = (argc > 1) ? argv[1] : "../parameters/theta_cn.prm";
    handler.parse_input(prm_file);
    prm.parse(handler);   // also maps CN/BE/FE -> Theta + theta value

    if (prm.dimension == 2)
      run_solver<2>(prm);
    else if (prm.dimension == 3)
      run_solver<3>(prm);
    else
      throw std::runtime_error("Dimension must be 2 or 3, got "
                                + std::to_string(prm.dimension));

  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
  return 0;
}