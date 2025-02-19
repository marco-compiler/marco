add_mlir_dialect_library(MLIRModelica
        Attributes.cpp
        AttrInterfaces.cpp
        Dialect.cpp
        Ops.cpp
        OpInterfaces.cpp
        Properties.cpp
        Types.cpp
        TypeInterfaces.cpp

        DEPENDS
        MLIRModelicaIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTransformUtils)
