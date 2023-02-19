set(SOURCES
    Lowering/AlgorithmLowerer.cpp
    Lowering/ArrayGeneratorLowerer.cpp
    Lowering/AssignmentStatementLowerer.cpp
    Lowering/BreakStatementLowerer.cpp
    Lowering/CallLowerer.cpp
    Lowering/ClassDependencyGraph.cpp
    Lowering/ClassPath.cpp
    Lowering/ClassLowerer.cpp
    Lowering/ConstantLowerer.cpp
    Lowering/ExpressionLowerer.cpp
    Lowering/EquationLowerer.cpp
    Lowering/ForStatementLowerer.cpp
    Lowering/IfStatementLowerer.cpp
    Lowering/Lowerer.cpp
    Lowering/LoweringContext.cpp
    Lowering/ModelLowerer.cpp
    Lowering/OperationLowerer.cpp
    Lowering/PackageLowerer.cpp
    Lowering/PartialDerFunctionLowerer.cpp
    Lowering/Reference.cpp
        Lowering/ComponentReferenceLowerer.cpp
    Lowering/Results.cpp
    Lowering/RecordLowerer.cpp
    Lowering/ReturnStatementLowerer.cpp
    Lowering/StandardFunctionLowerer.cpp
    Lowering/StatementLowerer.cpp
    Lowering/TupleLowerer.cpp
    Lowering/Verifier.cpp
    Lowering/WhenStatementLowerer.cpp
    Lowering/WhileStatementLowerer.cpp
    Bridge.cpp
    Runtime.cpp
    Options.cpp
    Utils.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

marco_add_library(codegen
    ${SOURCES}

    LINK_LIBS PUBLIC
    ginac
    marco::ast
    marco::modeling
    marco::variableFilter
    ${MLIR_LIBS})

add_subdirectory(Conversion)
add_subdirectory(Transforms)
