set(SOURCES
    VariableAccessAnalysis.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

add_mlir_dialect_library(MLIRBaseModelicaAnalysis
    ${SOURCES}

    LINK_LIBS PUBLIC
    marco::modeling
    ${MLIR_LIBS})
