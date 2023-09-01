add_mlir_dialect(Modeling modeling)
add_mlir_doc(Modeling Modeling Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Modeling.td)
mlir_tablegen(ModelingAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ModelingAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRModelingAttributesIncGen)
add_dependencies(MLIRModelingIncGen MLIRModelingAttributesIncGen)
