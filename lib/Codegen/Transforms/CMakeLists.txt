set(SOURCES
    ModelSolving/Access.cpp
    ModelSolving/CyclesLinearSolver.cpp
    ModelSolving/Equation.cpp
    ModelSolving/LoopEquation.cpp
    ModelSolving/Matching.cpp
    ModelSolving/Model.cpp
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
    ${SOURCES}

    DEPENDS
    marco::modeling

    LINK_LIBS PUBLIC
    marco::modeling
    ${MLIR_LIBS})
