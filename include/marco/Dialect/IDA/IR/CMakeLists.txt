add_mlir_doc(Dialect IDA Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRIDAAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=ida)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=ida)
add_public_tablegen_target(MLIRIDATypesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIDAOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ida)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=ida)
add_public_tablegen_target(MLIRIDAIncGen)

add_dependencies(MLIRIDAIncGen MLIRIDAAttributesIncGen)
add_dependencies(MLIRIDAIncGen MLIRIDATypesIncGen)
add_dependencies(MLIRIDAIncGen MLIRIDAOpsIncGen)

add_mlir_doc(Attributes IDAAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(Types IDATypes Dialects/ -gen-typedef-doc)
add_mlir_doc(Ops IDAOps Dialects/ -gen-op-doc)
