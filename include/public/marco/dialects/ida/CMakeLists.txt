add_mlir_dialect(IDA ida)
add_mlir_doc(IDA IDA Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS IDA.td)
mlir_tablegen(IDAAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(IDAAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRIDAAttributesIncGen)
add_dependencies(MLIRIDAIncGen MLIRIDAAttributesIncGen)
