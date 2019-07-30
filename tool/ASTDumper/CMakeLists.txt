add_executable(ASTDumper src/Main.cpp)
add_executable(modelica::ASTDumper ALIAS ASTDumper)
target_link_libraries(ASTDumper PUBLIC modelica::frontend)
target_compile_features(ASTDumper PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS ASTDumper RUNTIME DESTINATION bin)
