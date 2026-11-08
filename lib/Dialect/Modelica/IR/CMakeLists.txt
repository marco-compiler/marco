add_mlir_dialect_library(MLIRModelica
    Attributes.cpp
    AttrInterfaces.cpp
    ConstantFolding.cpp
    Dialect.cpp
    ElementOps.cpp
    Enums.cpp
    EquationOps.cpp
    InstanceOps.cpp
    Ops.cpp
    OpInterfaces.cpp
    OpUtils.cpp
    Properties.cpp
    ReferenceOps.cpp
    TypeInference.cpp
    Types.cpp
    TypeInterfaces.cpp

    DEPENDS
    MLIRModelicaIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils)
