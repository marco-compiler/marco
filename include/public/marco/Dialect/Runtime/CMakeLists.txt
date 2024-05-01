add_mlir_dialect(Runtime runtime)
add_mlir_doc(Runtime Runtime Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Runtime.td)
mlir_tablegen(RuntimeAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(RuntimeAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRRuntimeAttributesIncGen)
add_dependencies(MLIRRuntimeIncGen MLIRRuntimeAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS RuntimeInterfaces.td)
mlir_tablegen(RuntimeInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(RuntimeInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRRuntimeInterfacesIncGen)
add_dependencies(MLIRRuntimeIncGen MLIRRuntimeInterfacesIncGen)
