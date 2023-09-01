add_mlir_dialect_library(MLIRModeling
    Attributes.cpp
    ModelingDialect.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRModelingIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    marco::modeling)
