add_mlir_dialect_library(MLIRKINSOL
    Attributes.cpp
    KINSOLDialect.cpp
    Interfaces.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRKINSOLIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces)
