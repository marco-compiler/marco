add_mlir_dialect_library(MLIRBaseModelicaTransforms
    AutomaticDifferentiation/AutomaticDifferentiation.cpp
    AutomaticDifferentiation/ForwardAD.cpp
    Modeling/Bridge.cpp
    Modeling/EquationBridge.cpp
    Modeling/SCCBridge.cpp
    Modeling/VariableBridge.cpp
    Solvers/SUNDIALS.cpp
    AccessReplacementTest.cpp
    AllInterfaces.cpp
    AllocationOpInterfaceImpl.cpp
    BindingEquationConversion.cpp
    BufferizableOpInterfaceImpl.cpp
    CallCSE.cpp
    ConstantMaterializableTypeInterfaceImpl.cpp
    DerivableOpInterfaceImpl.cpp
    DerivableTypeInterfaceImpl.cpp
    DerivativeChainRule.cpp
    DerivativesAllocation.cpp
    DerivativesInitialization.cpp
    EquationAccessSplit.cpp
    EquationExplicitation.cpp
    EquationExplicitationTest.cpp
    EquationExpressionOpInterfaceImpl.cpp
    EquationFunctionLoopHoisting.cpp
    EquationInductionsExplicitation.cpp
    EquationSidesSplit.cpp
    EquationTemplatesCreation.cpp
    EulerForward.cpp
    ExplicitCastInsertion.cpp
    ExplicitInitialEquationsInsertion.cpp
    ExplicitStartValueInsertion.cpp
    FunctionDefaultValuesConversion.cpp
    FunctionScalarization.cpp
    FunctionUnwrap.cpp
    IDA.cpp
    InitialConditionsSolving.cpp
    InliningAttributeInsertion.cpp
    InvertibleOpInterfaceImpl.cpp
    Matching.cpp
    ModelAlgorithmConversion.cpp
    ModelDebugCanonicalization.cpp
    OpDistribution.cpp
    Passes.cpp
    PrintModelInfo.cpp
    PureFunctionInlining.cpp
    RangeBoundariesInference.cpp
    ReadOnlyVariablesPropagation.cpp
    RecordInlining.cpp
    RungeKutta.cpp
    RuntimeVerifiableOpInterfaceImpl.cpp
    ScalarRangesEquationSplit.cpp
    SCCAbsenceVerification.cpp
    SCCDetection.cpp
    SCCSolvingBySubstitution.cpp
    SCCSolvingWithKINSOL.cpp
    ScheduleParallelization.cpp
    SchedulersInstantiation.cpp
    Scheduling.cpp
    SingleValuedInductionElimination.cpp
    VariablesPromotion.cpp
    VariablesPruning.cpp
    VectorizableOpInterfaceImpl.cpp
    ViewAccessFolding.cpp

#    ADDITIONAL_HEADER_DIRS
#    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

    DEPENDS
    MLIRBaseModelicaTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRBaseModelica
    MLIRBaseModelicaAnalysis)
