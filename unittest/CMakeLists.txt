add_custom_target(MARCOUnitTests)
set_target_properties(MARCOUnitTests PROPERTIES FOLDER "MARCO unit tests")

include(GoogleTest)

add_subdirectory(Lexer)
add_subdirectory(Modeling)
add_subdirectory(Parser)
add_subdirectory(ParserError)
add_subdirectory(VariableFilter)
