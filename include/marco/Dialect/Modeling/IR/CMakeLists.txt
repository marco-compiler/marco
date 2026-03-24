set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRModelingAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRModelingOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=modeling)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=modeling)
add_public_tablegen_target(MLIRModelingIncGen)

add_dependencies(MLIRModelingIncGen MLIRModelingAttributesIncGen)
add_dependencies(MLIRModelingIncGen MLIRModelingOpsIncGen)

add_mlir_doc(Dialect Modeling Dialects/ -gen-dialect-doc)
add_mlir_doc(Attributes ModelingAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(Ops ModelingOps Dialects/ -gen-op-doc)
