modelicaAddLibrary(
        mlirlowerer SHARED
        src/ops/BasicOps.cpp
        src/ops/MathOps.cpp
        src/passes/patterns/BasicOpsPatterns.cpp
        src/passes/patterns/MathOpsPatterns.cpp
        src/passes/LowerModelica.cpp
        src/passes/LowerToLLVM.cpp
        src/passes/BreakRemover.cpp
        src/passes/ReturnRemover.cpp
        src/passes/TypeConverter.cpp
        src/Attribute.cpp
        src/MlirLowerer.cpp
        src/ModelicaBuilder.cpp
        src/ModelicaDialect.cpp
        src/Ops.cpp
        src/Type.cpp
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
