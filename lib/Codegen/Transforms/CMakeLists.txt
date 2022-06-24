set(SOURCES
    ModelSolving/ExternalSolvers/ExternalSolver.cpp
    ModelSolving/ExternalSolvers/IDA.cpp
    AutomaticDifferentiation/AutomaticDifferentiation.cpp
    AutomaticDifferentiation/ForwardAD.cpp
    ModelSolving/Access.cpp
    ModelSolving/ArrayVariable.cpp
    ModelSolving/Cycles.cpp
    ModelSolving/CyclesLinearSolver.cpp
    ModelSolving/DerivativesMap.cpp
    ModelSolving/Equation.cpp
    ModelSolving/FilteredVariable.cpp
    ModelSolving/LoopEquation.cpp
    ModelSolving/Matching.cpp
    ModelSolving/Model.cpp
    ModelSolving/ModelConverter.cpp
    ModelSolving/ModelSolving.cpp
    ModelSolving/Path.cpp
    ModelSolving/ScalarEquation.cpp
    ModelSolving/ScalarVariable.cpp
    ModelSolving/Scheduling.cpp
    ModelSolving/TypeConverter.cpp
    ModelSolving/Variable.cpp
    ModelSolving/VariablesMap.cpp
    ArrayDeallocation.cpp
    ExplicitCastInsertion.cpp
    FunctionScalarization.cpp
    Matching.cpp
    OpDistribution.cpp
    Scheduling.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs})

add_mlir_dialect_library(MLIRModelicaTransforms
    SHARED
    ${SOURCES}

    DEPENDS
    MARCOConversionPassIncGen
    MARCOTransformsPassIncGen

    LINK_LIBS PUBLIC
    marco::codegen
    marco::modeling
    MLIRModelicaCommonConversion
    ${MLIR_LIBS})
