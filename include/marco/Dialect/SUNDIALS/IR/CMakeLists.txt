set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSUNDIALSAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=sundials)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=sundials)
add_public_tablegen_target(MLIRSUNDIALSTypesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSUNDIALSOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=sundials)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=sundials)
add_public_tablegen_target(MLIRSUNDIALSIncGen)

add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSAttributesIncGen)
add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSTypesIncGen)
add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSOpsIncGen)

add_mlir_doc(Dialect SUNDIALS Dialects/ -gen-dialect-doc)
add_mlir_doc(Attributes SUNDIALSAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(Ops SUNDIALSOps Dialects/ -gen-op-doc)
