set(SOURCES
        EngineBuilder.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

marco_add_library(JIT
        ${SOURCES}

        LINK_COMPONENTS
        Core
        Support

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRExecutionEngine
        ${dialect_libs}
        ${extension_libs}
        ${conversion_libs})
