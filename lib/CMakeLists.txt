# Order matters.
# For example, the dialects library populate the global property
# containing the dialects list. Such property is then used by
# the Codegen and Frontend libraries.

# The modeling library is designed to be independent.
add_subdirectory(Modeling)

add_subdirectory(VariableFilter)
add_subdirectory(Diagnostic)
add_subdirectory(AST)
add_subdirectory(Parser)
add_subdirectory(Dialect)
add_subdirectory(Codegen)
add_subdirectory(Options)
add_subdirectory(IO)
add_subdirectory(Driver)
add_subdirectory(Frontend)

# Add the shared runtime support library for JIT compilation.
set(RUNTIME_SRC_ROOT ${CMAKE_SOURCE_DIR}/runtime/lib)

marco_add_runtime_library(RuntimeSupportShared SHARED
    ${RUNTIME_SRC_ROOT}/Support/BuiltInFunctions.cpp
    ${RUNTIME_SRC_ROOT}/Support/Math.cpp
    ${RUNTIME_SRC_ROOT}/Support/MemoryManagement.cpp
    ${RUNTIME_SRC_ROOT}/Support/Print.cpp
    ${RUNTIME_SRC_ROOT}/Support/UtilityFunctions.cpp)
