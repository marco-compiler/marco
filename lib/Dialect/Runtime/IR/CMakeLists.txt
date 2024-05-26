add_mlir_dialect_library(MLIRRuntime
    Attributes.cpp
    Interfaces.cpp
    RuntimeDialect.cpp
    Ops.cpp
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
