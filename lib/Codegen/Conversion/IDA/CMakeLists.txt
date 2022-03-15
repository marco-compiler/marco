set(SOURCES
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

add_mlir_dialect_library(MLIRIDAConversion
        SHARED
        ${SOURCES}

        DEPENDS
        marco::codegen # remove once dialects library is completed

        LINK_LIBS PUBLIC
        marco::codegen
        ${MLIR_LIBS})
