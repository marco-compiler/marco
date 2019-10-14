add_executable(simc src/Main.cpp)
add_executable(modelica::simc ALIAS simc)

target_link_libraries(simc PUBLIC modelica::simulation modelica::lowerer)
target_compile_features(simc PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS simc RUNTIME DESTINATION bin)

add_subdirectory(test)
