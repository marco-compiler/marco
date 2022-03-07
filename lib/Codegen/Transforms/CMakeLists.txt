set(SOURCES
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
    ModelSolving/Variable.cpp
    AutomaticDifferentiation.cpp
    BufferDeallocation.cpp
    ExplicitCastInsertion.cpp
    FunctionsVectorization.cpp
    OutputArraysPromotion.cpp)

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
    marco::codegen # remove once dialects library is completed
    marco::modeling

    LINK_LIBS PUBLIC
    marco::codegen
    marco::modeling
    ${MLIR_LIBS})
