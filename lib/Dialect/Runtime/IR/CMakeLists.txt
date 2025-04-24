add_mlir_dialect_library(MLIRRuntime
    Attributes.cpp
    Ops.cpp
    Properties.cpp
    Runtime.cpp
    Types.cpp

    DEPENDS
    MLIRRuntimeIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFunctionInterfaces
    MLIRSupport
    MLIRModeling)
