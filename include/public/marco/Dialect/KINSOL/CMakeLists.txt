add_mlir_dialect(KINSOL kinsol)
add_mlir_doc(KINSOL KINSOL Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS KINSOL.td)
mlir_tablegen(KINSOLAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(KINSOLAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRKINSOLAttributesIncGen)
add_dependencies(MLIRKINSOLIncGen MLIRKINSOLAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS KINSOLInterfaces.td)
mlir_tablegen(KINSOLInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(KINSOLInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRKINSOLInterfacesIncGen)
add_dependencies(MLIRKINSOLIncGen MLIRKINSOLInterfacesIncGen)
