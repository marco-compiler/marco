modelicaAddLibrary(
        mlirlowerer SHARED
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
        src/ModelicaToStandard.cpp
        src/Ops.cpp
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
