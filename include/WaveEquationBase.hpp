#pragma once

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
    void run();

  protected:
    virtual void advance_one_step() = 0;
    virtual void init_scheme_state() {}
    virtual std::string scheme_name() const = 0;

    // ── Adaptive time stepping (optional, used by WaveRK45) ──
    // Default: fixed dt, existing 4 schemes unaffected.
    virtual bool is_adaptive() const { return false; }
    // Advance one step with embedded error estimate; on return,
    // `dt` may have been updated for the *next* step, and the
    // function returns the *accepted* dt used for this step
    // (0.0 if the step was rejected and must be retried).
    virtual double advance_one_step_adaptive() { return 0.0; }

    void make_grid();
    void setup_system();
    void assemble_matrices();

    void compute_acceleration(TrilinosWrappers::MPI::Vector &acc,
                               const TrilinosWrappers::MPI::Vector &u) const;
    void apply_dirichlet_bc(TrilinosWrappers::MPI::Vector &v, double t) const;

    double compute_energy()            const;
    double compute_l2_error (double t) const;
    double compute_h1_error (double t) const;   // NEW: H1 seminorm
    void   output_vtu(unsigned int step)  const;

    // MMS: add source term to RHS vector (called by WaveTheta)
    void add_mms_source(TrilinosWrappers::MPI::Vector &rhs, double t) const;

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

    TrilinosWrappers::MPI::Vector solution;
    TrilinosWrappers::MPI::Vector velocity;
    TrilinosWrappers::MPI::Vector acceleration;

    double       time;
    double       dt;
    unsigned int step_number;

    // Profiling
    mutable TimerOutput timer_output;

  private:
    std::ofstream energy_log;
    std::ofstream error_log;
    std::ofstream adaptive_log;
  };

} // namespace WaveSolver

#include "WaveEquationBase.tpp"