add_mlir_dialect_library(MLIRSBG
    Attributes.cpp
    SBGDialect.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRSBGIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport)

target_link_libraries(MLIRSBG PUBLIC sbg-util)
target_link_libraries(MLIRSBG PUBLIC sbg)
target_link_libraries(MLIRSBG PUBLIC boost)