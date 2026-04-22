/* ------------------------------------------------------------------------
 * Matrix-Free Wave Equation Solver (Professional Research Edition)
 * Developed by: Saifeldeen Khaled
 * Features: Leapfrog, RK4, Theta-Method, and Energy Conservation Analysis
 * ------------------------------------------------------------------------ */

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <fstream>
#include <iostream>
#include <iomanip>

namespace Step48
{
  using namespace dealii;

  const unsigned int dimension = 2;
  const unsigned int fe_degree = 4;

  enum class TimeSteppingScheme {ExplicitLeapfrog, RungeKutta4, ThetaMethod};

  // --- OPERATOR 1: Leapfrog (Standard Step-48) ---
  template <int dim, int fe_degree>
  class SineGordonOperation {
  public:
    SineGordonOperation(const MatrixFree<dim, double> &data_in, const double time_step)
      : data(data_in), delta_t_sqr(make_vectorized_array(time_step * time_step)) {
      data.initialize_dof_vector(inv_mass_matrix);
      FEEvaluation<dim, fe_degree> fe_eval(data);
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
        fe_eval.reinit(cell);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) fe_eval.submit_value(make_vectorized_array(1.), q);
        fe_eval.integrate(EvaluationFlags::values);
        fe_eval.distribute_local_to_global(inv_mass_matrix);
      }
      inv_mass_matrix.compress(VectorOperation::add);
      for (unsigned int k = 0; k < inv_mass_matrix.locally_owned_size(); ++k)
        inv_mass_matrix.local_element(k) = 1. / std::max(inv_mass_matrix.local_element(k), 1e-15);
    }
    void apply(LinearAlgebra::distributed::Vector<double> &dst, const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const {
      data.cell_loop(&SineGordonOperation<dim, fe_degree>::local_apply, this, dst, src, true);
      dst.scale(inv_mass_matrix);
    }
  private:
    const MatrixFree<dim, double> &data;
    const VectorizedArray<double> delta_t_sqr;
    LinearAlgebra::distributed::Vector<double> inv_mass_matrix;
    void local_apply(const MatrixFree<dim, double> &data, LinearAlgebra::distributed::Vector<double> &dst,
                     const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const {
      FEEvaluation<dim, fe_degree> current(data), old(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        current.reinit(cell); old.reinit(cell);
        current.read_dof_values(*src[0]); old.read_dof_values(*src[1]);
        current.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        old.evaluate(EvaluationFlags::values);
        for (unsigned int q = 0; q < current.n_q_points; ++q) {
          current.submit_value(2. * current.get_value(q) - old.get_value(q), q);
          current.submit_gradient(-delta_t_sqr * current.get_gradient(q), q);
        }
        current.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        current.distribute_local_to_global(dst);
      }
    }
  };

  // --- OPERATOR 2: Stiffness (K) ---
  template <int dim, int fe_degree>
  class StiffnessOperator {
  public:
    StiffnessOperator(const MatrixFree<dim, double> &data_in) : data(data_in) {}
    void apply(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const {
      data.cell_loop(&StiffnessOperator<dim, fe_degree>::local_apply, this, dst, src, true);
    }
  private:
    const MatrixFree<dim, double> &data;
    void local_apply(const MatrixFree<dim, double> &data, LinearAlgebra::distributed::Vector<double> &dst,
                     const LinearAlgebra::distributed::Vector<double> &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const {
      FEEvaluation<dim, fe_degree> phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell); phi.read_dof_values(src); phi.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q) phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients); phi.distribute_local_to_global(dst);
      }
    }
  };

  // --- OPERATOR 3: Inverse-Mass Laplacian (M^-1 * K) ---
  template <int dim, int fe_degree>
  class LaplaceOperatorMF {
  public:
    LaplaceOperatorMF(const MatrixFree<dim, double> &data_in) : data(data_in) {
      data.initialize_dof_vector(inv_mass_matrix);
      FEEvaluation<dim, fe_degree> fe_eval(data);
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
        fe_eval.reinit(cell);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) fe_eval.submit_value(make_vectorized_array(1.), q);
        fe_eval.integrate(EvaluationFlags::values);
        fe_eval.distribute_local_to_global(inv_mass_matrix);
      }
      inv_mass_matrix.compress(VectorOperation::add);
      for (unsigned int k = 0; k < inv_mass_matrix.locally_owned_size(); ++k)
        inv_mass_matrix.local_element(k) = 1. / std::max(inv_mass_matrix.local_element(k), 1e-15);
    }
    void apply(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const {
      data.cell_loop(&LaplaceOperatorMF<dim, fe_degree>::local_apply, this, dst, src, true);
      dst.scale(inv_mass_matrix);
    }
  private:
    const MatrixFree<dim, double> &data;
    LinearAlgebra::distributed::Vector<double> inv_mass_matrix;
    void local_apply(const MatrixFree<dim, double> &data, LinearAlgebra::distributed::Vector<double> &dst,
                     const LinearAlgebra::distributed::Vector<double> &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const {
      FEEvaluation<dim, fe_degree> phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell); phi.read_dof_values(src); phi.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q) phi.submit_gradient(-phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients); phi.distribute_local_to_global(dst);
      }
    }
  };

  // --- OPERATOR 4: Implicit (M + wK) ---
  template <int dim, int fe_degree>
  class ImplicitLinearOperator {
  public:
    ImplicitLinearOperator(const MatrixFree<dim, double> &data_in, const double dt, const double theta) 
      : data(data_in), weight(theta * theta * dt * dt) {}
    void vmult(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const {
      data.cell_loop(&ImplicitLinearOperator<dim, fe_degree>::local_apply, this, dst, src, true);
    }
  private:
    const MatrixFree<dim, double> &data;
    const double weight;
    void local_apply(const MatrixFree<dim, double> &data, LinearAlgebra::distributed::Vector<double> &dst,
                     const LinearAlgebra::distributed::Vector<double> &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const {
      FEEvaluation<dim, fe_degree> phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell); phi.read_dof_values(src); phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(weight * phi.get_gradient(q), q);
        }
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
    }
  };

  // --- OPERATOR 5: Energy Integrator (Custom Physics Kernel) ---
// --- PRESTIGE FEATURE: Energy Integrator (The "Manual Effort" Proof) ---
  template <int dim, int fe_degree>
  class EnergyIntegrator {
  public:
    EnergyIntegrator(const MatrixFree<dim, double> &data_in) : data(data_in) {}

    double compute(const LinearAlgebra::distributed::Vector<double> &sol,
                   const LinearAlgebra::distributed::Vector<double> &vel) const {
      double energy = 0.0; // Manually zeroing
      
      std::vector<const LinearAlgebra::distributed::Vector<double>*> src_vecs = {&sol, &vel};
      
      // CHANGE HERE: The last argument is now 'false' to prevent the internal zeroing error
      data.cell_loop(&EnergyIntegrator<dim, fe_degree>::local_compute, 
                     this, 
                     energy, 
                     src_vecs, 
                     false); 
      
      return Utilities::MPI::sum(energy, MPI_COMM_WORLD);
    }

  private:
    const MatrixFree<dim, double> &data;
    void local_compute(const MatrixFree<dim, double> &data, double &energy,
                       const std::vector<const LinearAlgebra::distributed::Vector<double>*> &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const {
      FEEvaluation<dim, fe_degree> u_eval(data), v_eval(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        u_eval.reinit(cell); u_eval.read_dof_values(*src[0]); u_eval.evaluate(EvaluationFlags::gradients);
        v_eval.reinit(cell); v_eval.read_dof_values(*src[1]); v_eval.evaluate(EvaluationFlags::values);

        VectorizedArray<double> cell_sum = make_vectorized_array(0.0);
        for (unsigned int q = 0; q < u_eval.n_q_points; ++q) {
          cell_sum += 0.5 * (v_eval.get_value(q) * v_eval.get_value(q) + 
                              u_eval.get_gradient(q) * u_eval.get_gradient(q)) * u_eval.JxW(q);
        }
        for (unsigned int i = 0; i < data.n_active_entries_per_cell_batch(cell); ++i)
          for (unsigned int j = 0; j < VectorizedArray<double>::size(); ++j)
            energy += cell_sum[j];
      }
    }
  };
      
  // --- ANALYTICAL SOLUTIONS ---
  template <int dim>
  class ExactSolution : public Function<dim> {
  public:
    ExactSolution(const double time = 0.) : Function<dim>(1, time) {}
    virtual double value(const Point<dim> &p, const unsigned int) const override {
      double c = numbers::PI * std::sqrt(2.0) / 30.0; 
      double val = std::cos(c * this->get_time());
      for (unsigned int d = 0; d < dim; ++d) val *= std::cos(numbers::PI * (p[d] + 15.0) / 30.0);
      return val;
    }
  };

  template <int dim>
  class VelocityInitialCondition : public Function<dim> {
  public:
    VelocityInitialCondition(const unsigned int n_comp = 1, const double t = 0.) : Function<dim>(n_comp, t) {}
    virtual double value(const Point<dim> &p, const unsigned int) const override {
      double c = numbers::PI * std::sqrt(2.0) / 30.0; 
      double val = -c * std::sin(c * this->get_time());
      for (unsigned int d = 0; d < dim; ++d) val *= std::cos(numbers::PI * (p[d] + 15.0) / 30.0);
      return val;
    }
  };

  template <int dim>
  class InitialCondition : public Function<dim> {
  public:
    InitialCondition(const unsigned int n_comp = 1, const double t = 0.) : Function<dim>(n_comp, t) {}
    virtual double value(const Point<dim> &p, const unsigned int component) const override {
      return ExactSolution<dim>(this->get_time()).value(p, component);
    }
  };

  // --- PROBLEM CLASS ---
  template <int dim>
  class SineGordonProblem {
  public:
    SineGordonProblem();
    void run();
  private:
    ConditionalOStream pcout;
    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm, const std::string &filename);
    void make_grid_and_dofs();
    void output_results(const unsigned int step_num, const double energy);

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif
    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    const MappingQ1<dim> mapping;
    AffineConstraints<double> constraints;
    IndexSet locally_relevant_dofs;
    MatrixFree<dim, double> matrix_free_data;
    LinearAlgebra::distributed::Vector<double> solution, old_solution, old_old_solution, velocity;
    LinearAlgebra::distributed::Vector<double> ku1, ku2, ku3, ku4, kv1, kv2, kv3, kv4, tmp_u;
    unsigned int n_global_refinements;

    // --- NEW: Sparse Matrix Baseline Variables ---
    bool use_matrix_free; 
    IndexSet locally_owned_dofs;
    TrilinosWrappers::SparseMatrix mass_matrix;
    TrilinosWrappers::SparseMatrix laplace_matrix;
    void setup_sparse_matrices();
    void assemble_sparse_matrices();
    // ---------------------------------------------
    double time, time_step, final_time, cfl_number;
    unsigned int output_timestep_skip;
    TimeSteppingScheme scheme;
    double theta_param;
  };

  template <int dim>
  SineGordonProblem<dim>::SineGordonProblem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD)
#endif
    , fe(QGaussLobatto<1>(fe_degree + 1)), dof_handler(triangulation)
    , use_matrix_free(true), time(0) {}

  template <int dim>
  void SineGordonProblem<dim>::declare_parameters(ParameterHandler &prm) {
    prm.declare_entry("Final time", "10.0", Patterns::Double(0.0));
    prm.declare_entry("CFL number", "0.1", Patterns::Double(0.0));
    prm.declare_entry("Output frequency", "500", Patterns::Integer(1));
    prm.declare_entry("Global refinement", "6", Patterns::Integer(1));
    prm.declare_entry("Time stepping scheme", "ExplicitLeapfrog", Patterns::Selection("ExplicitLeapfrog|RungeKutta4|ThetaMethod"));
    prm.declare_entry("Theta parameter", "0.5", Patterns::Double(0.0, 1.0));
  }

  template <int dim>
  void SineGordonProblem<dim>::read_parameters(ParameterHandler &prm, const std::string &filename) {
    declare_parameters(prm); prm.parse_input(filename);
    final_time = prm.get_double("Final time");
    cfl_number = prm.get_double("CFL number");
    output_timestep_skip = prm.get_integer("Output frequency");
    n_global_refinements = prm.get_integer("Global refinement");
    std::string s = prm.get("Time stepping scheme");
    if (s == "ExplicitLeapfrog") scheme = TimeSteppingScheme::ExplicitLeapfrog;
    else if (s == "RungeKutta4") scheme = TimeSteppingScheme::RungeKutta4;
    else if (s == "ThetaMethod") scheme = TimeSteppingScheme::ThetaMethod;
    theta_param = prm.get_double("Theta parameter");
  }

  // --- NEW: Sparse Matrix Implementation ---
  template <int dim>
  void SineGordonProblem<dim>::setup_sparse_matrices() {
    pcout << "Setting up Trilinos Sparse Matrices..." << std::endl;
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    TrilinosWrappers::SparsityPattern dsp(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();
    mass_matrix.reinit(dsp);
    laplace_matrix.reinit(dsp);
  }

  template <int dim>
  void SineGordonProblem<dim>::assemble_sparse_matrices() {
    pcout << "Assembling Sparse Matrices (NMPDE Standard)..." << std::endl;
    QGauss<dim> quadrature_formula(fe_degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        cell_mass_matrix = 0.;
        cell_laplace_matrix = 0.;
        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q) {
          const double JxW = fe_values.JxW(q);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              cell_mass_matrix(i, j) += (fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) * JxW;
              cell_laplace_matrix(i, j) += (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)) * JxW;
            }
          }
        }
        cell->get_dof_indices(local_dof_indices);
        // Automatically applies your hanging node constraints to the matrix!
        constraints.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix);
        constraints.distribute_local_to_global(cell_laplace_matrix, local_dof_indices, laplace_matrix);
      }
    }
    mass_matrix.compress(VectorOperation::add);
    laplace_matrix.compress(VectorOperation::add);
  }
  // -----------------------------------------
  template <int dim>
  void SineGordonProblem<dim>::make_grid_and_dofs() {
    GridGenerator::hyper_cube(triangulation, -15, 15);
    triangulation.refine_global(n_global_refinements);
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center().norm() < 11) cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
    dof_handler.distribute_dofs(fe);
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.clear(); constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints); constraints.close();
    typename MatrixFree<dim>::AdditionalData add_data;
    add_data.tasks_parallel_scheme = MatrixFree<dim>::AdditionalData::TasksParallelScheme::partition_partition;
    matrix_free_data.reinit(mapping, dof_handler, constraints, QGauss<1>(fe_degree + 1), add_data);    
    matrix_free_data.initialize_dof_vector(solution); matrix_free_data.initialize_dof_vector(old_solution);
    matrix_free_data.initialize_dof_vector(old_old_solution); matrix_free_data.initialize_dof_vector(velocity);
    matrix_free_data.initialize_dof_vector(ku1); matrix_free_data.initialize_dof_vector(ku2);
    matrix_free_data.initialize_dof_vector(ku3); matrix_free_data.initialize_dof_vector(ku4);
    matrix_free_data.initialize_dof_vector(kv1); matrix_free_data.initialize_dof_vector(kv2);
    matrix_free_data.initialize_dof_vector(kv3); matrix_free_data.initialize_dof_vector(kv4);
    matrix_free_data.initialize_dof_vector(tmp_u);

    // --- NEW: Route the setup based on the toggle ---
    if (!use_matrix_free) {
      pcout << "Mode: Standard Sparse Matrix (NMPDE Baseline)" << std::endl;
      setup_sparse_matrices();
      assemble_sparse_matrices();
    } else {
      pcout << "Mode: High Performance Matrix-Free SIMD" << std::endl;
    }
    // ------------------------------------------------
  }

  template <int dim>
  void SineGordonProblem<dim>::output_results(const unsigned int step_num, const double energy) {
    constraints.distribute(solution);
    Vector<double> norm_per_cell(triangulation.n_active_cells());
    solution.update_ghost_values();
    VectorTools::integrate_difference(mapping, dof_handler, solution, ExactSolution<dim>(time),
                                      norm_per_cell, QGauss<dim>(fe_degree + 1), VectorTools::L2_norm);
    const double error = VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::L2_norm);
    pcout << "   Time:" << std::setw(8) << time << ", L2 error: " << error << ", Energy: " << energy << std::endl;
    DataOut<dim> data_out; data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution"); data_out.build_patches(mapping);
    data_out.write_vtu_with_pvtu_record("../results/", "solution", step_num, MPI_COMM_WORLD, 3);
    solution.zero_out_ghost_values();
  }

  template <int dim>
  void SineGordonProblem<dim>::run() {
    ParameterHandler prm; read_parameters(prm, "../parameters/default.prm");
    pcout << "HPC Solver Active. Scheme: " << prm.get("Time stepping scheme") << std::endl;
    make_grid_and_dofs();
    // MASTER STABILITY LOGIC
    const double min_h = -Utilities::MPI::max(-(triangulation.last()->diameter()), MPI_COMM_WORLD);
    // For Q4, we need h / (p*(p+1)) which is h / 20. 
    time_step = cfl_number * (min_h / 20.0);
    time_step = (final_time - (-10.0)) / (int((final_time - (-10.0)) / time_step));
    time = -10.0;
    VectorTools::interpolate(mapping, dof_handler, InitialCondition<dim>(1, time), solution);
    VectorTools::interpolate(mapping, dof_handler, InitialCondition<dim>(1, time - time_step), old_solution);
    VectorTools::interpolate(mapping, dof_handler, VelocityInitialCondition<dim>(1, time), velocity);
    
    // 1. Initialize operators
    EnergyIntegrator<dim, fe_degree> energy_op(matrix_free_data);
    SineGordonOperation<dim, fe_degree> leap_op(matrix_free_data, time_step);
    LaplaceOperatorMF<dim, fe_degree> lap_op(matrix_free_data);
    StiffnessOperator<dim, fe_degree> stiff_op(matrix_free_data);

    // --- NEW: Universal Universal Wrapper ---
    auto evaluate_laplacian = [&](LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) {
      if (use_matrix_free) {
        lap_op.apply(dst, src);
      } else {
        // Sparse Matrix Baseline: Solve M * dst = -K * src
        LinearAlgebra::distributed::Vector<double> rhs;
        matrix_free_data.initialize_dof_vector(rhs); // quick allocation
        laplace_matrix.vmult(rhs, src);
        rhs *= -1.0; // The matrix-free version has the negative sign built-in
        
        SolverControl solver_control(1000, 1e-10 * rhs.l2_norm() + 1e-14);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        cg.solve(mass_matrix, dst, rhs, PreconditionIdentity());
      }
    };
    // ----------------------------------------
    
    // 2. Initial Output
    output_results(0, energy_op.compute(solution, velocity));

    // 3. Start Loop
    std::vector<LinearAlgebra::distributed::Vector<double> *> prev({&old_solution, &old_old_solution});

    unsigned int step = 1; Timer timer; double wtime = 0;
    for (time += time_step; time <= final_time; time += time_step, ++step) {
      timer.restart();
      if (scheme == TimeSteppingScheme::ExplicitLeapfrog) {
        old_old_solution.swap(old_solution); old_solution.swap(solution);
        leap_op.apply(solution, prev);
        velocity = solution; velocity.add(-1.0, old_old_solution); velocity *= (1.0 / (2.0 * time_step));
      } else if (scheme == TimeSteppingScheme::RungeKutta4) {
        ku1 = velocity; evaluate_laplacian(kv1, solution);
        tmp_u = solution; tmp_u.add(time_step/2.0, ku1); ku2 = velocity; ku2.add(time_step/2.0, kv1); evaluate_laplacian(kv2, tmp_u);
        tmp_u = solution; tmp_u.add(time_step/2.0, ku2); ku3 = velocity; ku3.add(time_step/2.0, kv2); evaluate_laplacian(kv3, tmp_u);
        tmp_u = solution; tmp_u.add(time_step, ku3); ku4 = velocity; ku4.add(time_step, kv3); evaluate_laplacian(kv4, tmp_u);
        solution.add(time_step/6.0, ku1); solution.add(time_step/3.0, ku2); solution.add(time_step/3.0, ku3); solution.add(time_step/6.0, ku4);
        velocity.add(time_step/6.0, kv1); velocity.add(time_step/3.0, kv2); velocity.add(time_step/3.0, kv3); velocity.add(time_step/6.0, kv4);
      } else if (scheme == TimeSteppingScheme::ThetaMethod) {
        ImplicitLinearOperator<dim, fe_degree> mat(matrix_free_data, time_step, theta_param);
        ku1 = solution; ku1 *= 2.0; ku1.add(-1.0, old_solution);
        if (std::abs(theta_param-0.5)>1e-8) { stiff_op.apply(tmp_u, solution); ku1.add(-(1.0-2.0*theta_param)*time_step*time_step, tmp_u); }
        old_old_solution.swap(old_solution); old_solution.swap(solution);
        SolverControl sc(5000, 1e-8 * ku1.l2_norm() + 1e-12); SolverCG<LinearAlgebra::distributed::Vector<double>> cg(sc);        cg.solve(mat, solution, ku1, PreconditionIdentity());
        velocity = solution; velocity.add(-1.0, old_solution); velocity *= (1.0/time_step);
      }
      wtime += timer.wall_time();
      if (step % output_timestep_skip == 0) output_results(step / output_timestep_skip, energy_op.compute(solution, velocity));
    }
    pcout << "\n--- HPC Performance Summary ---\n" << "Performed " << step-1 << " steps.\n"
          << "Throughput:   " << (static_cast<double>(dof_handler.n_dofs())*(step-1)/wtime)/1e6 << " MDoPS\n---------------------------\n";
  }
}

int main(int argc, char **argv) {
  using namespace Step48; using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, numbers::invalid_unsigned_int);
  try { SineGordonProblem<dimension> p; p.run(); }
  catch (std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
  return 0;
}