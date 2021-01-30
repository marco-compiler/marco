modelicaAddLibrary(
        mlirlowerer SHARED
        src/BreakRemover.cpp
        src/LowerToLLVM.cpp
        src/LowerToStandard.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
        src/Ops.cpp
        src/ReturnRemover.cpp
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
