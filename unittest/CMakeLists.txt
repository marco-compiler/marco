add_custom_target(MARCOUnitTests)
set_target_properties(MARCOUnitTests PROPERTIES FOLDER "MARCO unit tests")

add_subdirectory(ast)
add_subdirectory(codegen)
add_subdirectory(modeling)
add_subdirectory(runtime)
add_subdirectory(utils)
