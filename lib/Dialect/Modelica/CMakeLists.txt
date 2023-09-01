add_mlir_dialect_library(MLIRModelica
    Attributes.cpp
    EquationPath.cpp
    ExpressionPath.cpp
    ModelicaDialect.cpp
    Interfaces.cpp
    Ops.cpp
    Types.cpp
    VariableAccess.cpp

    DEPENDS
    MLIRModelicaIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
    MLIRModeling
    MLIRTransformUtils
    marco::modeling)
