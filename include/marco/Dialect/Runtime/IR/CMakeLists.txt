set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRRuntimeAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRuntimeOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=runtime)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=runtime)
add_public_tablegen_target(MLIRRuntimeTypesIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=runtime)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=runtime)
add_public_tablegen_target(MLIRRuntimeIncGen)

add_dependencies(MLIRRuntimeIncGen MLIRRuntimeAttributesIncGen)
add_dependencies(MLIRRuntimeIncGen MLIRRuntimeTypesIncGen)
add_dependencies(MLIRRuntimeIncGen MLIRRuntimeOpsIncGen)

add_mlir_doc(Dialect Runtime Dialects/ -gen-dialect-doc)
add_mlir_doc(Attributes RuntimeAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(Ops RuntimeOps Dialects/ -gen-op-doc)
