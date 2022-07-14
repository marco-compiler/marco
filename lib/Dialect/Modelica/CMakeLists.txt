add_mlir_dialect_library(MLIRModelica
    Attributes.cpp
    ModelicaDialect.cpp
    Interfaces.cpp
    Ops.cpp
    Shape.cpp
    Types.cpp

    DEPENDS
    MLIRModelicaIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces)
