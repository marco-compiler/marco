set(SOURCES
    BaseModelicaToTensor.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    ${conversion_libs})

add_mlir_conversion_library(MLIRBaseModelicaToTensor
    ${SOURCES}

    DEPENDS
    MARCOConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRBaseModelicaCommonConversion
    ${MLIR_LIBS})
