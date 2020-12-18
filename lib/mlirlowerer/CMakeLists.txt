modelicaAddLibrary(
        mlirlowerer SHARED
        src/CmpOps.cpp
        src/LLVMLoweringPass.cpp
        src/MathOps.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
        src/ModelicaToStandard.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        modelica::frontend
        modelica::utils
        MLIRIR
        MLIRSCF
        MLIRSCFToStandard
        MLIRStandard
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRExecutionEngine
)
