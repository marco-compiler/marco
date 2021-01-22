modelicaAddLibrary(
        mlirlowerer SHARED
        src/BreakRemover.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
        src/ModelicaLoweringPass.cpp
        src/Ops.cpp
        src/ReturnRemover.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        modelica::frontend
        modelica::utils
        MLIRIR
        MLIRLinalg
        MLIRLinalgToStandard
        MLIRLinalgToLLVM
        MLIROpenMP
        MLIROpenMPToLLVM
        MLIRSCF
        MLIRSCFToStandard
        MLIRSCFToOpenMP
        MLIRStandard
        MLIRStandardToLLVM
        MLIRLinalgTransforms
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRExecutionEngine
)
