set(SOURCES
    ModelicaToBaseModelica.cpp)

add_mlir_conversion_library(MLIRModelicaToBaseModelica
    ${SOURCES}

    DEPENDS
    MARCOConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRBaseModelica
    MLIRModelica
    MLIRModelicaTransforms)
