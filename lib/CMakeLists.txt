# Order matters.
# For example, the dialects library populate the global property
# containing the dialects list. Such property is then used by
# the Codegen and Frontend libraries.

# The modeling library is designed to be independent.
add_subdirectory(Modeling)

add_subdirectory(VariableFilter)
add_subdirectory(Lexer)
add_subdirectory(AST)
add_subdirectory(Parser)
add_subdirectory(Dialect)
add_subdirectory(Codegen)
add_subdirectory(JIT)
add_subdirectory(IO)
add_subdirectory(Frontend)
add_subdirectory(Transforms)
