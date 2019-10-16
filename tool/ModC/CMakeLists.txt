add_executable(modc src/Main.cpp)
add_executable(modelica::modc ALIAS modc)

target_link_libraries(modc PUBLIC modelica::model modelica::lowerer)
target_compile_features(modc PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS modc RUNTIME DESTINATION bin)

add_subdirectory(test)
