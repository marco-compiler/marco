get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_dialect_library(MLIRModelicaTransforms
    Evaluation.cpp
    ExpandConnections.cpp
    Flatten.cpp
    Instantiate.cpp
    Instantiation/Expressions.cpp
    Instantiation/InstanceTree.cpp
    Instantiation/Lookup.cpp
    Instantiation/Model.cpp
    Passes.cpp

    DEPENDS
    MLIRModelicaTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRModelica
    MLIRTransformUtils
    ${dialect_libs})
