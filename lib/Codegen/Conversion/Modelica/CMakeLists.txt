set(SOURCES
    LowerToCFG.cpp
    LowerToLLVM.cpp
    ModelicaConversion.cpp
    TypeConverter.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    ${conversion_libs})

add_mlir_dialect_library(MLIRModelicaConversion
    SHARED
    ${SOURCES}

    DEPENDS
    marco::codegen
    marco::modeling
    MLIRIDAConversion

    LINK_LIBS PUBLIC
    marco::codegen
    marco::modeling
    MLIRIDAConversion
    ${MLIR_LIBS})
