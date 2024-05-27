add_mlir_doc(IDA IDA Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS IDAAttributes.td)
mlir_tablegen(IDAAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(IDAAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRIDAAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS IDATypes.td)
mlir_tablegen(IDATypes.h.inc -gen-typedef-decls -typedefs-dialect=ida)
mlir_tablegen(IDATypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ida)
add_public_tablegen_target(MLIRIDATypesIncGen)

set(LLVM_TARGET_DEFINITIONS IDAOps.td)
mlir_tablegen(IDAOps.h.inc -gen-op-decls)
mlir_tablegen(IDAOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIDAOpsIncGen)

set(LLVM_TARGET_DEFINITIONS IDA.td)
mlir_tablegen(IDA.h.inc -gen-dialect-decls -dialect=ida)
mlir_tablegen(IDA.cpp.inc -gen-dialect-defs -dialect=ida)
add_public_tablegen_target(MLIRIDAIncGen)

add_dependencies(MLIRIDAIncGen MLIRIDAAttributesIncGen)
add_dependencies(MLIRIDAIncGen MLIRIDATypesIncGen)
add_dependencies(MLIRIDAIncGen MLIRIDAOpsIncGen)
