modelicaAddLibrary(
        mlirlowerer SHARED
        src/LLVMLoweringPass.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        modelica::frontend
        modelica::utils
        MLIRIR
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTransforms
)
