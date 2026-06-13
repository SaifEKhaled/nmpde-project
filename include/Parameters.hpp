#pragma once
/* ============================================================
 * Parameters.hpp
 * ============================================================ */
#include <deal.II/base/parameter_handler.h>
#include <string>

struct Parameters {
  double       final_time     = 1.0;
  double       wave_speed     = 1.0;
  unsigned int refinements    = 5;
  unsigned int fe_degree      = 1;
  std::string  scheme         = "CN";     // CN | BE | FE | Leapfrog | RK4
  double       theta          = 0.5;      // used only by Theta scheme
  double       dt             = 0.01;
  unsigned int output_every   = 20;
  std::string  output_dir     = "../results";

  void declare(dealii::ParameterHandler &prm) {
    prm.declare_entry("Final time",   "1.0",      dealii::Patterns::Double(0));
    prm.declare_entry("Wave speed",   "1.0",      dealii::Patterns::Double(0));
    prm.declare_entry("Refinements",  "5",        dealii::Patterns::Integer(1));
    prm.declare_entry("FE degree",    "1",        dealii::Patterns::Integer(1));
    prm.declare_entry("Scheme",       "CN",
      dealii::Patterns::Selection("CN|BE|FE|Leapfrog|RK4"));
    prm.declare_entry("Theta",        "0.5",      dealii::Patterns::Double(0,1));
    prm.declare_entry("Time step",    "0.01",     dealii::Patterns::Double(0));
    prm.declare_entry("Output every", "20",       dealii::Patterns::Integer(0));
    prm.declare_entry("Output dir",   "../results",dealii::Patterns::Anything());
  }

  void parse(dealii::ParameterHandler &prm) {
    final_time   = prm.get_double("Final time");
    wave_speed   = prm.get_double("Wave speed");
    refinements  = prm.get_integer("Refinements");
    fe_degree    = prm.get_integer("FE degree");
    scheme       = prm.get("Scheme");
    theta        = prm.get_double("Theta");
    dt           = prm.get_double("Time step");
    output_every = prm.get_integer("Output every");
    output_dir   = prm.get("Output dir");

    // Map friendly names to theta values
    if      (scheme == "CN") { scheme = "Theta"; theta = 0.5; }
    else if (scheme == "BE") { scheme = "Theta"; theta = 1.0; }
    else if (scheme == "FE") { scheme = "Theta"; theta = 0.0; }
  }
};