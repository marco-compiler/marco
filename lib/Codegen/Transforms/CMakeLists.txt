set(SOURCES
    ModelSolving/ExternalSolvers/ExternalSolver.cpp
    ModelSolving/ExternalSolvers/IDA.cpp
    AutomaticDifferentiation/AutomaticDifferentiation.cpp
    AutomaticDifferentiation/ForwardAD.cpp
    ModelSolving/Access.cpp
    ModelSolving/Cycles.cpp
    ModelSolving/CyclesLinearSolver.cpp
    ModelSolving/Equation.cpp
    ModelSolving/LoopEquation.cpp
    ModelSolving/Matching.cpp
    ModelSolving/Model.cpp
    ModelSolving/ModelConverter.cpp
    ModelSolving/ModelSolving.cpp
    ModelSolving/Path.cpp
    ModelSolving/ScalarEquation.cpp
    ModelSolving/Scheduling.cpp
    ModelSolving/TypeConverter.cpp
    ModelSolving/Variable.cpp
    ArrayDeallocation.cpp
    ExplicitCastInsertion.cpp
    FunctionScalarization.cpp
    OpDistribution.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    MLIRIR
    MLIRSCF
    MLIRStandard
    MLIRTransforms
    MLIRTranslation)

add_mlir_dialect_library(MLIRModelicaTransforms
    SHARED
    ${SOURCES}

    DEPENDS
    MLIRIDAConversion
    MLIRModelicaConversion
    marco::codegen # remove once dialects library is completed
    marco::modeling

    LINK_LIBS PUBLIC
    marco::codegen
    marco::modeling
    MLIRIDAConversion
    MLIRModelicaConversion
    ${MLIR_LIBS})
