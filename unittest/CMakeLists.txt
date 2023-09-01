add_custom_target(MARCOUnitTests)
set_target_properties(MARCOUnitTests PROPERTIES FOLDER "MARCO unit tests")

include(GoogleTest)

add_subdirectory(AST)
add_subdirectory(Codegen)
add_subdirectory(Dialect)
add_subdirectory(Modeling)
add_subdirectory(Parser)
add_subdirectory(VariableFilter)
