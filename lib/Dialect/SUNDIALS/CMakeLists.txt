add_mlir_dialect_library(MLIRSUNDIALS
    Attributes.cpp
    SUNDIALSDialect.cpp
    Interfaces.cpp
    Ops.cpp
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
