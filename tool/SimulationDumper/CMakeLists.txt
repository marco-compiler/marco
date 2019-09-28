add_executable(SimulationDumper src/Main.cpp)
add_executable(modelica::SimulationDumper ALIAS SimulationDumper)
target_link_libraries(SimulationDumper PUBLIC modelica::lowerer)
target_compile_features(SimulationDumper PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS SimulationDumper RUNTIME DESTINATION bin)

add_subdirectory(test)
