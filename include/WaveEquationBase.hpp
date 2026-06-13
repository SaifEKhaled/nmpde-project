#pragma once
/* ============================================================
 * WaveEquationBase.hpp
 * Abstract base class for the 2-D wave equation FEM solver.
 *
 * Provides:
 *   - Mesh creation & DoF setup
 *   - Mass / stiffness matrix assembly
 *   - M*acc = -c^2*K*u  acceleration solve
 *   - Energy, L2 error, VTU output, CSV logging
 *
 * Derived classes (WaveTheta, WaveLeapfrog, WaveRK4) implement
 * advance_one_step() and are responsible for maintaining
 * solution / velocity / acceleration vectors.
 * ============================================================ */

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/distributed/tria.h>

#include "Parameters.hpp"
#include "WaveExact.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>

namespace WaveSolver {
  using namespace dealii;

  template <int dim>
  class WaveEquationBase {
  public:
    WaveEquationBase(const Parameters &prm);
    virtual ~WaveEquationBase() = default;

    // Full time-integration loop — calls advance_one_step() each step
    void run();

  protected:
    // ── Override in derived classes ──────────────────────────
    virtual void advance_one_step() = 0;
    // Called once after initial conditions are set
    virtual void init_scheme_state() {}
    // Human-readable scheme name for console output
    virtual std::string scheme_name() const = 0;

    // ── Helpers available to derived classes ────────────────
    void make_grid();
    void setup_system();
    void assemble_matrices();

    // Solve M*acc = -c^2 * K * u
    void compute_acceleration(TrilinosWrappers::MPI::Vector &acc,
                               const TrilinosWrappers::MPI::Vector &u) const;

    // Zero the boundary DOFs of v (exact = 0 for our test case)
    void apply_dirichlet_bc(TrilinosWrappers::MPI::Vector &v, double t) const;

    double compute_energy()   const;
    double compute_l2_error(double t) const;
    void   output_vtu(unsigned int step) const;

    // ── Data ────────────────────────────────────────────────
    const Parameters &prm;
    ConditionalOStream pcout;
    MPI_Comm           mpi_comm;

    parallel::distributed::Triangulation<dim> triangulation;
    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;
    MappingQ1<dim>  mapping;
    AffineConstraints<double> constraints;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    TrilinosWrappers::SparseMatrix mass_matrix;
    TrilinosWrappers::SparseMatrix stiffness_matrix;

    TrilinosWrappers::MPI::Vector solution;      // u^n
    TrilinosWrappers::MPI::Vector velocity;      // du/dt^n
    TrilinosWrappers::MPI::Vector acceleration;  // d²u/dt²^n  (carried between steps)

    double       time;
    double       dt;
    unsigned int step_number;

  private:
    std::ofstream energy_log;
    std::ofstream error_log;
  };

} // namespace WaveSolver

#include "WaveEquationBase.tpp"
