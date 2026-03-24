add_mlir_dialect_library(MLIRIDA
    Attributes.cpp
    Dialect.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRIDAIncGen

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
