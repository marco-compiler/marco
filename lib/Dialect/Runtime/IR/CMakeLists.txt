add_mlir_dialect_library(MLIRRuntime
    Attributes.cpp
    Ops.cpp
    Runtime.cpp

    DEPENDS
    MLIRRuntimeIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFunctionInterfaces
    MLIRSupport
    MLIRModeling)
