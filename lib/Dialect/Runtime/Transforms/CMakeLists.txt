get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_dialect_library(MLIRRuntimeTransforms
    AllInterfaces.cpp
    BufferizableOpInterfaceImpl.cpp
    Passes.cpp

    DEPENDS
    MLIRRuntimeTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRRuntime
    ${dialect_libs})
