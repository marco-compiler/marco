# Order matters.
# For example, the dialects library populate the global property
# containing the dialects list. Such property is then used by
# the Codegen and Frontend libraries.

# The runtime library is independent, so its relative position
# doesn't matter.
add_subdirectory(Runtime)

# Also the modeling library is designed to be independent.
add_subdirectory(Modeling)

add_subdirectory(Utils)
add_subdirectory(VariableFilter)
add_subdirectory(Diagnostic)
add_subdirectory(AST)
add_subdirectory(Dialect)
add_subdirectory(Codegen)
add_subdirectory(Frontend)
