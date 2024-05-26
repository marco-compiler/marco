add_mlir_dialect(SUNDIALS sundials)
add_mlir_doc(SUNDIALS SUNDIALS Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS SUNDIALS.td)
mlir_tablegen(SUNDIALSAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SUNDIALSAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSUNDIALSAttributesIncGen)
add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SUNDIALSInterfaces.td)
mlir_tablegen(SUNDIALSInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SUNDIALSInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRSUNDIALSInterfacesIncGen)
add_dependencies(MLIRSUNDIALSIncGen MLIRSUNDIALSInterfacesIncGen)
