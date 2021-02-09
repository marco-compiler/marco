modelicaAddLibrary(
        mlirlowerer SHARED
        src/ops/BasicOps.cpp
        src/ops/LogicOps.cpp
        src/ops/LoopOps.cpp
        src/ops/MathOps.cpp
        src/passes/patterns/BasicOpsPatterns.cpp
        src/passes/patterns/LogicOpsPatterns.cpp
        src/passes/patterns/LoopOpsPatterns.cpp
        src/passes/patterns/MathOpsPatterns.cpp
        src/passes/LowerModelica.cpp
        src/passes/LowerToLLVM.cpp
        src/passes/BreakRemover.cpp
        src/passes/ReturnRemover.cpp
        src/ArrayType.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        modelica::frontend
        modelica::utils
        MLIRIR
        MLIRAffineToStandard
        MLIRLinalg
        MLIRLinalgToStandard
        MLIRSCF
        MLIRSCFToStandard
        MLIRStandard
        MLIRStandardToLLVM
        MLIRVectorToLLVM
        MLIRLinalgTransforms
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRExecutionEngine
)
