set(SOURCES
    LowerToCFG.cpp
    LowerToLLVM.cpp
    ModelicaConversion.cpp
    TypeConverter.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    MLIRIR
    MLIRMathToLLVM
    MLIROpenMPToLLVM
    MLIROpenMPToLLVMIRTranslation
    MLIRSCF
    MLIRSCFToStandard
    MLIRSCFToOpenMP
    MLIRStandard
    MLIRStandardToLLVM
    MLIRTransforms
    MLIRTranslation)

add_mlir_dialect_library(MLIRModelicaConversion
    ${SOURCES}

    DEPENDS
    marco::modeling

    LINK_LIBS PUBLIC
    marco::modeling
    ${MLIR_LIBS})
