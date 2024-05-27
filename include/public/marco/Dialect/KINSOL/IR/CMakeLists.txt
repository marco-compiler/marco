add_mlir_doc(KINSOL KINSOL Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS KINSOLAttributes.td)
mlir_tablegen(KINSOLAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(KINSOLAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRKINSOLAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS KINSOLTypes.td)
mlir_tablegen(KINSOLTypes.h.inc -gen-typedef-decls -typedefs-dialect=kinsol)
mlir_tablegen(KINSOLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=kinsol)
add_public_tablegen_target(MLIRKINSOLTypesIncGen)

set(LLVM_TARGET_DEFINITIONS KINSOLOps.td)
mlir_tablegen(KINSOLOps.h.inc -gen-op-decls)
mlir_tablegen(KINSOLOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRKINSOLOpsIncGen)

set(LLVM_TARGET_DEFINITIONS KINSOL.td)
mlir_tablegen(KINSOL.h.inc -gen-dialect-decls -dialect=kinsol)
mlir_tablegen(KINSOL.cpp.inc -gen-dialect-defs -dialect=kinsol)
add_public_tablegen_target(MLIRKINSOLIncGen)

add_dependencies(MLIRKINSOLIncGen MLIRKINSOLAttributesIncGen)
add_dependencies(MLIRKINSOLIncGen MLIRKINSOLTypesIncGen)
add_dependencies(MLIRKINSOLIncGen MLIRKINSOLOpsIncGen)
