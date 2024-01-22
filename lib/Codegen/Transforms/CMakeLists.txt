set(SOURCES
    AutomaticDifferentiation/AutomaticDifferentiation.cpp
    AutomaticDifferentiation/ForwardAD.cpp
    Modeling/Bridge.cpp
    Modeling/EquationBridge.cpp
    Modeling/MatchedEquationBridge.cpp
    Modeling/SCCBridge.cpp
    Modeling/VariableBridge.cpp
    Solvers/SUNDIALS.cpp
    AccessReplacementTest.cpp
    ArrayDeallocation.cpp
    BindingEquationConversion.cpp
    DerivativesAllocation.cpp
    EquationAccessSplit.cpp
    EquationExplicitation.cpp
    EquationExplicitationTest.cpp
    EquationFunctionLoopHoisting.cpp
    EquationInductionsExplicitation.cpp
    EquationSidesSplit.cpp
    EquationTemplatesCreation.cpp
    EulerForward.cpp
    ExplicitCastInsertion.cpp
    ExplicitInitialEquationsInsertion.cpp
    ExplicitStartValueInsertion.cpp
    FunctionInlining.cpp
    FunctionScalarization.cpp
    IDA.cpp
    InitialConditionsSolving.cpp
    Matching.cpp
    ModelAlgorithmConversion.cpp
    ModelDebugCanonicalization.cpp
    OpDistribution.cpp
    RangeBoundariesInference.cpp
    ReadOnlyVariablesPropagation.cpp
    RecordInlining.cpp
    SCCAbsenceVerification.cpp
    SCCDetection.cpp
    SCCSolvingBySubstitution.cpp
    SCCSolvingWithKINSOL.cpp
    ScheduleParallelization.cpp
    SchedulersInstantiation.cpp
    Scheduling.cpp
    VariablesPromotion.cpp
    ViewAccessFolding.cpp)

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
