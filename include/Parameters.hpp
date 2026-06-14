#pragma once
#include <deal.II/base/parameter_handler.h>
#include <string>

struct Parameters {
  double       final_time     = 1.0;
  double       wave_speed     = 1.0;
  unsigned int dimension       = 2;       // 2 or 3
  unsigned int refinements    = 5;
  unsigned int fe_degree      = 1;
  std::string  scheme         = "CN";     // CN | BE | FE | Leapfrog | RK4
  double       theta          = 0.5;
  double       dt             = 0.01;
  unsigned int output_every   = 20;
  std::string  output_dir     = "../results";
  bool         use_mms        = false;    // manufactured solution test
  bool         profiling      = false;    // print TimerOutput breakdown

  void declare(dealii::ParameterHandler &prm) {
    prm.declare_entry("Final time",   "1.0",       dealii::Patterns::Double(0));
    prm.declare_entry("Wave speed",   "1.0",       dealii::Patterns::Double(0));
    prm.declare_entry("Dimension",    "2",         dealii::Patterns::Integer(2,3));
    prm.declare_entry("Refinements",  "5",         dealii::Patterns::Integer(1));
    prm.declare_entry("FE degree",    "1",         dealii::Patterns::Integer(1));
    prm.declare_entry("Scheme",       "CN",
      dealii::Patterns::Selection("CN|BE|FE|Leapfrog|RK4"));
    prm.declare_entry("Theta",        "0.5",       dealii::Patterns::Double(0,1));
    prm.declare_entry("Time step",    "0.01",      dealii::Patterns::Double(0));
    prm.declare_entry("Output every", "20",        dealii::Patterns::Integer(0));
    prm.declare_entry("Output dir",   "../results",dealii::Patterns::Anything());
    prm.declare_entry("MMS",          "false",     dealii::Patterns::Bool());
    prm.declare_entry("Profiling",    "false",     dealii::Patterns::Bool());
  }

  void parse(dealii::ParameterHandler &prm) {
    final_time   = prm.get_double  ("Final time");
    wave_speed   = prm.get_double  ("Wave speed");
    dimension    = prm.get_integer ("Dimension");
    refinements  = prm.get_integer ("Refinements");
    fe_degree    = prm.get_integer ("FE degree");
    scheme       = prm.get         ("Scheme");
    theta        = prm.get_double  ("Theta");
    dt           = prm.get_double  ("Time step");
    output_every = prm.get_integer ("Output every");
    output_dir   = prm.get         ("Output dir");
    use_mms      = prm.get_bool    ("MMS");
    profiling    = prm.get_bool    ("Profiling");

    if      (scheme == "CN") { scheme = "Theta"; theta = 0.5; }
    else if (scheme == "BE") { scheme = "Theta"; theta = 1.0; }
    else if (scheme == "FE") { scheme = "Theta"; theta = 0.0; }
  }
};