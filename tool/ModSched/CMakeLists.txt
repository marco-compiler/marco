add_executable(modsched src/Main.cpp)
add_executable(modelica::modsched ALIAS modsched)

target_link_libraries(modsched PUBLIC modelica::model modelica::passes)
target_compile_features(modsched PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS modsched RUNTIME DESTINATION bin)

add_subdirectory(test)
