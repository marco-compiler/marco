set(SOURCES
    AutomaticDifferentiation/AutomaticDifferentiation.cpp
    AutomaticDifferentiation/ForwardAD.cpp
    ModelSolving/Access.cpp
    ModelSolving/ArrayVariable.cpp
    ModelSolving/Cycles.cpp
    ModelSolving/CyclesLinearSolver.cpp
    ModelSolving/CyclesSymbolicSolver.cpp
    ModelSolving/DerivativesMap.cpp
    ModelSolving/Equation.cpp
    ModelSolving/FilteredVariable.cpp
    ModelSolving/LoopEquation.cpp
    ModelSolving/Matching.cpp
    ModelSolving/Model.cpp
    ModelSolving/Path.cpp
    ModelSolving/ScalarEquation.cpp
    ModelSolving/ScalarVariable.cpp
    ModelSolving/Scheduling.cpp
    ModelSolving/TypeConverter.cpp
    ModelSolving/Utils.cpp
    ModelSolving/Variable.cpp
    ArrayDeallocation.cpp
    CyclesSolving.cpp
    EulerForward.cpp
    ExplicitCastInsertion.cpp
    FunctionInlining.cpp
    FunctionScalarization.cpp
    IDA.cpp
    Matching.cpp
    ModelLegalization.cpp
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
    MLIRModelicaCommonConversion
    MLIRIDACommonConversion
    ${MLIR_LIBS})
