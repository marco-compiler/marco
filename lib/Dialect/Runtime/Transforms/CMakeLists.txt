add_mlir_dialect_library(MLIRRuntimeTransforms
    AllInterfaces.cpp
    BufferizableOpInterfaceImpl.cpp
    Passes.cpp

    #    ADDITIONAL_HEADER_DIRS
    #    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

    DEPENDS
    MLIRRuntimeTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRRuntime)
