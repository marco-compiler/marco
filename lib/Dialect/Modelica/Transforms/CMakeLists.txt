get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_dialect_library(MLIRModelicaTransforms
    ConnectionExpansion.cpp
    ConnectionExpansion/AccessFunctionUtils.cpp
    ConnectionExpansion/ConnectionEquations.cpp
    ConnectionExpansion/ConnectionGraph.cpp
    Evaluation.cpp
    Flattening.cpp
    Instantiation.cpp
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
