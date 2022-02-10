add_mlir_dialect_library(MLIRModelica SHARED
    ModelicaAttributes.cpp
    ModelicaDialect.cpp
    ModelicaOps.cpp
    ModelicaTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${MARCO_INCLUDE_PUBLIC_DIR}/marco/dialects/modelica

    DEPENDS
    MLIRModelicaIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces)
