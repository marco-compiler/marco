add_mlir_dialect_library(MLIRModelica
    Attributes.cpp
    AttrInterfaces.cpp
    EquationPath.cpp
    ExpressionPath.cpp
    ModelicaDialect.cpp
    Ops.cpp
    OpInterfaces.cpp
    Types.cpp
    TypeInterfaces.cpp
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
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
    MLIRModeling
    MLIRTransformUtils
    marco::modeling)
