add_mlir_dialect_library(MLIRRuntimeTransforms
    AllInterfaces.cpp
    BufferizableOpInterfaceImpl.cpp
    Passes.cpp

    DEPENDS
    MLIRRuntimeTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRRuntime)
