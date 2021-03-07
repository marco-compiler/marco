modelicaAddLibrary(
        mlirlowerer SHARED
        src/passes/LowerToLLVM.cpp
        src/passes/BreakRemover.cpp
        src/passes/ReturnRemover.cpp
        src/passes/TypeConverter.cpp
        src/Attribute.cpp
        src/MlirLowerer.cpp
        src/ModelicaBuilder.cpp
        src/ModelicaDialect.cpp
        src/Ops.cpp
        src/Runner.cpp
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
        MLIRSCFToGPU
        MLIRStandard
        MLIRStandardToLLVM
        MLIRVectorToLLVM
        MLIRLinalgTransforms
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRExecutionEngine
)
