add_mlir_dialect_library(MLIRModelicaTransforms
        Passes.cpp

        DEPENDS
        MLIRModelicaTransformsIncGen

        LINK_LIBS PUBLIC
        MLIRModelica)
