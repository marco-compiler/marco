add_mlir_dialect_library(MLIRModeling
    Attributes.cpp
    Dialect.cpp
    Ops.cpp
    Properties.cpp

    DEPENDS
    MLIRModelingIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    marco::modeling)
