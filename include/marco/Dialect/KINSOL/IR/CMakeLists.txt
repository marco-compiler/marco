set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRKINSOLAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=kinsol)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=kinsol)
add_public_tablegen_target(MLIRKINSOLTypesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRKINSOLOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=kinsol)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=kinsol)
add_public_tablegen_target(MLIRKINSOLIncGen)

add_dependencies(MLIRKINSOLIncGen MLIRKINSOLAttributesIncGen)
add_dependencies(MLIRKINSOLIncGen MLIRKINSOLTypesIncGen)
add_dependencies(MLIRKINSOLIncGen MLIRKINSOLOpsIncGen)

add_mlir_doc(Dialect KINSOL Dialects/ -gen-dialect-doc)
add_mlir_doc(Attributes KINSOLAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(Types KINSOLTypes Dialects/ -gen-typedef-doc)
add_mlir_doc(Ops KINSOLOps Dialects/ -gen-op-doc)
