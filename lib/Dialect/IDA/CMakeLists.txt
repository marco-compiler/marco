add_mlir_dialect_library(MLIRIDA
    Attributes.cpp
    IDADialect.cpp
    Interfaces.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRIDAIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces)
