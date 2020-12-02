modelicaAddLibrary(
        mlirlowerer SHARED
        src/FunctionLowerer.cpp
        src/FunctionOp.cpp
        src/MlirLowerer.cpp
        src/ModelicaDialect.cpp
)

target_link_libraries(
        mlirlowerer PUBLIC
        MLIRIR
        modelica::frontend
        modelica::utils
)
