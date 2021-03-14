modelicaAddLibrary(
        mlirlowerer SHARED
        src/passes/ModelicaConversionPass.cpp
        src/passes/LowerToLLVM.cpp
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
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation
        MLIRSCF
        MLIRSCFToStandard
        MLIRSCFToOpenMP
        MLIRSCFToGPU
        MLIRStandard
        MLIRStandardToLLVM
        MLIRVectorToLLVM
        MLIRTransforms
        MLIRTranslation
        MLIRExecutionEngine
)
