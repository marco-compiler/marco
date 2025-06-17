add_mlir_doc(Modeling Modeling Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ModelingAttributes.td)
mlir_tablegen(ModelingAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ModelingAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRModelingAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS ModelingOps.td)
mlir_tablegen(ModelingOps.h.inc -gen-op-decls)
mlir_tablegen(ModelingOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRModelingOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Modeling.td)
mlir_tablegen(Modeling.h.inc -gen-dialect-decls -dialect=modeling)
mlir_tablegen(Modeling.cpp.inc -gen-dialect-defs -dialect=modeling)
add_public_tablegen_target(MLIRModelingIncGen)

add_dependencies(MLIRModelingIncGen MLIRModelingAttributesIncGen)
add_dependencies(MLIRModelingIncGen MLIRModelingOpsIncGen)

add_mlir_doc(ModelingAttributes ModelingAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(ModelingOps ModelingOps Dialects/ -gen-op-doc)
