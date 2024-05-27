add_mlir_dialect_library(MLIRKINSOL
    Attributes.cpp
    KINSOL.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRKINSOLIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRModeling
    MLIRSupport
    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
    MLIRSUNDIALS)
