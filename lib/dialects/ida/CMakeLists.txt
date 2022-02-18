add_mlir_dialect_library(MLIRIDA SHARED
    Attributes.cpp
    IDADialect.cpp
    Ops.cpp
    Types.cpp

    ADDITIONAL_HEADER_DIRS
    ${MARCO_INCLUDE_PUBLIC_DIR}/marco/dialects/ida

    DEPENDS
    MLIRIDAIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces)
