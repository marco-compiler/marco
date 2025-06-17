add_mlir_doc(SUNDIALS SUNDIALS Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS SUNDIALSAttributes.td)
mlir_tablegen(SUNDIALSAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SUNDIALSAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSUNDIALSAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SUNDIALSTypes.td)
mlir_tablegen(SUNDIALSTypes.h.inc -gen-typedef-decls -typedefs-dialect=sundials)
mlir_tablegen(SUNDIALSTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=sundials)
add_public_tablegen_target(MLIRSUNDIALSTypesIncGen)

set(LLVM_TARGET_DEFINITIONS SUNDIALSOps.td)
mlir_tablegen(SUNDIALSOps.h.inc -gen-op-decls)
mlir_tablegen(SUNDIALSOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSUNDIALSOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SUNDIALS.td)
mlir_tablegen(SUNDIALS.h.inc -gen-dialect-decls -dialect=sundials)
mlir_tablegen(SUNDIALS.cpp.inc -gen-dialect-defs -dialect=sundials)
add_public_tablegen_target(MLIRSUNDIALSIncGen)

add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSAttributesIncGen)
add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSTypesIncGen)
add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSOpsIncGen)

add_mlir_doc(SUNDIALSAttributes SUNDIALSAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(SUNDIALSOps SUNDIALSOps Dialects/ -gen-op-doc)
