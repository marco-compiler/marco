modelicaAddLibrary(
        mlirlowerer SHARED
        src/AssignmentOp.cpp
        src/ConstantOpOld.cpp
        src/FunctionLowerer.cpp
        src/FunctionOp.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
        src/ReturnOp.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        MLIRIR
        MLIRStandardOps
        modelica::frontend
        modelica::utils
)
