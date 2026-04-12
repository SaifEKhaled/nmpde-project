cmake_minimum_required(VERSION 3.12)

project(wave_equation LANGUAGES CXX)

find_package(deal.II REQUIRED HINTS /usr/lib/x86_64-linux-gnu/cmake/deal.II)

deal_ii_initialize_cached_variables()

add_executable(wave_equation src/wave_equation.cc)

deal_ii_setup_target(wave_equation)
