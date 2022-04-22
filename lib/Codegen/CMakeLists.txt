set(SOURCES
    Lowering/CallLowerer.cpp
    Lowering/ClassLowerer.cpp
    Lowering/ExpressionLowerer.cpp
    Lowering/EquationLowerer.cpp
    Lowering/Lowerer.cpp
    Lowering/LoweringContext.cpp
    Lowering/ModelLowerer.cpp
    Lowering/OperationLowerer.cpp
    Lowering/Reference.cpp
    Lowering/Results.cpp
    Lowering/StandardFunctionLowerer.cpp
    Lowering/StatementLowerer.cpp
    ArrayDescriptor.cpp
    Bridge.cpp
    Reference.cpp
    Runtime.cpp
    Options.cpp
    Utils.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

marco_add_library(codegen
    SHARED
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::variableFilter
    marco::utils

    LINK_LIBS PUBLIC
    marco::ast
    marco::variableFilter
    marco::utils
    ${MLIR_LIBS})

add_subdirectory(Conversion)
add_subdirectory(Transforms)
