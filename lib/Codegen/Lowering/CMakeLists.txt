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
    ScopedSymbolTable.cpp
    StandardFunctionLowerer.cpp
    StatementLowerer.cpp
    SubscriptLowerer.cpp
    Symbol.cpp
    TupleLowerer.cpp
    WhenEquationLowerer.cpp
    WhenStatementLowerer.cpp
    WhileStatementLowerer.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

marco_add_library(Lowering
    ${SOURCES}

    LINK_LIBS PUBLIC
    marco::ASTBaseModelica
    marco::modeling
    marco::variableFilter
    marco::Lexer
    ${MLIR_LIBS})
