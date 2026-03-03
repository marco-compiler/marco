get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_dialect_library(MLIRModelicaTransforms
    Passes.cpp

    DEPENDS
    MLIRModelicaTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRModelica
    ${dialect_libs})
