add_mlir_doc(Runtime Runtime Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS RuntimeAttributes.td)
mlir_tablegen(RuntimeAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(RuntimeAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRRuntimeAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS RuntimeOps.td)
mlir_tablegen(RuntimeOps.h.inc -gen-op-decls)
mlir_tablegen(RuntimeOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRuntimeOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Runtime.td)
mlir_tablegen(Runtime.h.inc -gen-dialect-decls -dialect=runtime)
mlir_tablegen(Runtime.cpp.inc -gen-dialect-defs -dialect=runtime)
add_public_tablegen_target(MLIRRuntimeIncGen)

add_dependencies(MLIRRuntimeIncGen MLIRRuntimeAttributesIncGen)
add_dependencies(MLIRRuntimeIncGen MLIRRuntimeOpsIncGen)

add_mlir_doc(RuntimeAttributes RuntimeAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(RuntimeOps RuntimeOps Dialects/ -gen-op-doc)
