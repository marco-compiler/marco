# Order matters.
# For example, the dialects library populate the global property
# containing the dialects list. Such property is then used by
# the Codegen and Frontend libraries.

# The runtime library is independent, so its relative position
# doesn't matter.
add_subdirectory(runtime)

# Also the modeling library is designed to be independent.
add_subdirectory(modeling)

add_subdirectory(utils)
add_subdirectory(diagnostic)
add_subdirectory(ast)
add_subdirectory(dialects)
add_subdirectory(codegen)
add_subdirectory(frontend)
add_subdirectory(frontendTool)
