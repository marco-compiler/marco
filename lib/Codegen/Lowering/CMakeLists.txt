set(SOURCES
    AlgorithmLowerer.cpp
    ArrayGeneratorLowerer.cpp
    AssignmentStatementLowerer.cpp
    BreakStatementLowerer.cpp
    Bridge.cpp
    CallLowerer.cpp
    CallStatementLowerer.cpp
    ClassDependencyGraph.cpp
    ClassPath.cpp
    ClassLowerer.cpp
    ComponentReferenceLowerer.cpp
    ConstantLowerer.cpp
    EqualityEquationLowerer.cpp
    EquationLowerer.cpp
    EquationSectionLowerer.cpp
    ExpressionLowerer.cpp
    ForEquationLowerer.cpp
    ForStatementLowerer.cpp
    IdentifierError.cpp
    IfEquationLowerer.cpp
    IfStatementLowerer.cpp
    Lowerer.cpp
    LoweringContext.cpp
    ModelLowerer.cpp
    OperationLowerer.cpp
    PackageLowerer.cpp
    PartialDerFunctionLowerer.cpp
    Reference.cpp
    Results.cpp
    RecordLowerer.cpp
    ReturnStatementLowerer.cpp
    StandardFunctionLowerer.cpp
    StatementLowerer.cpp
    SubscriptLowerer.cpp
    TupleLowerer.cpp
    VariablesSymbolTable.cpp
    WhenEquationLowerer.cpp
    WhenStatementLowerer.cpp
    WhileStatementLowerer.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(Distance)

set(MLIR_LIBS
    ${dialect_libs})

marco_add_library(Lowering
    ${SOURCES}

    LINK_LIBS PUBLIC
    marco::ast
    marco::modeling
    marco::variableFilter
    marco::Distance
    ${MLIR_LIBS})
