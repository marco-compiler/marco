add_executable(modmatch src/Main.cpp)
add_executable(modelica::modmatch ALIAS modmatch)

target_link_libraries(modmatch PUBLIC modelica::model modelica::lowerer)
target_compile_features(modmatch PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS modmatch RUNTIME DESTINATION bin)

add_subdirectory(test)
