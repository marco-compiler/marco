set(SOURCES
    AutomaticDifferentiation/AutomaticDifferentiation.cpp
    AutomaticDifferentiation/ForwardAD.cpp
    AccessReplacementTest.cpp
    ArrayDeallocation.cpp
    BindingEquationConversion.cpp
    CyclesSolving.cpp
    DerivativesAllocation.cpp
    EquationAccessSplit.cpp
    EquationExplicitationTest.cpp
    EquationTemplatesCreation.cpp
    EquationViewsComputation.cpp
    EulerForward.cpp
    ExplicitCastInsertion.cpp
    ExplicitInitialEquationsInsertion.cpp
    ExplicitStartValueInsertion.cpp
    FunctionInlining.cpp
    FunctionScalarization.cpp
    IDA.cpp
    Matching.cpp
    ModelAlgorithmConversion.cpp
    OpDistribution.cpp
    ReadOnlyVariablesPropagation.cpp
    RecordInlining.cpp
    Scheduling.cpp
    SolverPassBase.cpp
    VariablesPromotion.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

add_mlir_dialect_library(MLIRModelicaTransforms
    ${SOURCES}

    DEPENDS
    MARCOConversionPassIncGen
    MARCOTransformsPassIncGen

    LINK_LIBS PUBLIC
    marco::codegen
    marco::modeling
    MLIRModelicaAnalysis
    MLIRModelicaCommonConversion
    MLIRIDACommonConversion
    ${MLIR_LIBS})
