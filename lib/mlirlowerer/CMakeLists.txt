modelicaAddLibrary(
        mlirlowerer SHARED
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        MLIRIR
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRTransforms
        modelica::frontend
        modelica::utils
)
