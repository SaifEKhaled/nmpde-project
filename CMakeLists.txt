cmake_minimum_required(VERSION 3.13)
project(WaveEquation CXX)

find_package(deal.II 9.3
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  REQUIRED)

deal_ii_initialize_cached_variables()

add_executable(wave_equation src/main.cc)
target_include_directories(wave_equation PRIVATE ${CMAKE_SOURCE_DIR}/include)
deal_ii_setup_target(wave_equation)

# Create results/ next to the project root (where the executable is run from)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)
