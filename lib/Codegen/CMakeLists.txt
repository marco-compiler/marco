set(SOURCES
    Runtime.cpp
    Verifier.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

marco_add_library(codegen
    ${SOURCES}

    LINK_LIBS PUBLIC
    ${MLIR_LIBS})

add_subdirectory(Conversion)
add_subdirectory(Lowering)
