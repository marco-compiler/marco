add_executable(omcc src/Main.cpp)
add_executable(modelica::omcc ALIAS omcc)

target_link_libraries(omcc PUBLIC modelica::omcToModel modelica::lowerer)
target_compile_features(omcc PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS omcc RUNTIME DESTINATION bin)

add_subdirectory(test)
