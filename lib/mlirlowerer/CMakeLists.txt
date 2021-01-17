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
        MLIRLinalg
        MLIRLinalgToStandard
        MLIROpenMP
        MLIROpenMPToLLVM
        MLIRSCFToStandard
        MLIRSCFToOpenMP
        MLIRStandard
        MLIRStandardToLLVM
        MLIRLinalgTransforms
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRExecutionEngine
)
