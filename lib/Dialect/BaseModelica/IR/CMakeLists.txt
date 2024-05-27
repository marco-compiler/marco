add_mlir_dialect_library(MLIRBaseModelica
    Attributes.cpp
    AttrInterfaces.cpp
    DefaultValuesDependencyGraph.cpp
    Dialect.cpp
    EquationPath.cpp
    ExpressionPath.cpp
    Ops.cpp
    OpInterfaces.cpp
    Types.cpp
    TypeInterfaces.cpp
    VariableAccess.cpp
    VariablesDependencyGraph.cpp
    VariablesDimensionsDependencyGraph.cpp

    DEPENDS
    MLIRBaseModelicaIncGen

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
