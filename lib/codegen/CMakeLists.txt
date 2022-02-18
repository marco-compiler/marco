set(SOURCES
    dialects/modelica/Attribute.cpp
    dialects/modelica/ModelicaBuilder.cpp
    dialects/modelica/ModelicaDialect.cpp
    dialects/modelica/Ops.cpp
    dialects/modelica/Type.cpp
    passes/AutomaticDifferentiation.cpp
    passes/BufferDeallocation.cpp
    passes/ExplicitCastInsertion.cpp
    passes/FunctionsVectorization.cpp
    passes/ModelicaConversion.cpp
    passes/LowerToCFG.cpp
    passes/LowerToLLVM.cpp
    passes/ResultBuffersToArgs.cpp
    passes/SolveModel.cpp
    passes/TypeConverter.cpp
    passes/model/Access.cpp
    passes/model/Equation.cpp
    passes/model/LinearSolver.cpp
    passes/model/Matching.cpp
    passes/model/Model.cpp
    passes/model/Path.cpp
    passes/model/ScalarEquation.cpp
    passes/model/Scheduling.cpp
    passes/model/LoopEquation.cpp
    passes/model/Variable.cpp
    passes/model/Cycles.cpp
    CodeGen.cpp
    Options.cpp
    Reference.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    MLIRIR
    MLIRMathToLLVM
    MLIROpenMPToLLVM
    MLIROpenMPToLLVMIRTranslation
    MLIRSCF
    MLIRSCFToStandard
    MLIRSCFToOpenMP
    MLIRStandard
    MLIRStandardToLLVM
    MLIRTransforms
    MLIRTranslation)

marco_add_library(codegen
    SHARED
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::modeling
    marco::utils

    LINK_LIBS
    PRIVATE marco::ast
    PRIVATE marco::modeling
    PRIVATE marco::utils
    PRIVATE ${MLIR_LIBS})
