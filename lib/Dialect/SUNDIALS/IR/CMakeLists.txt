add_mlir_dialect_library(MLIRSUNDIALS
    Attributes.cpp
    Ops.cpp
    SUNDIALS.cpp
    Types.cpp

    DEPENDS
    MLIRSUNDIALSIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRModeling
    MLIRSupport
    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces)
