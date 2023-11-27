add_mlir_dialect(SBG sbg)
add_mlir_doc(SBG SBG Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS SBG.td)
mlir_tablegen(SBGAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SBGAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSBGAttributesIncGen)
add_dependencies(MLIRSBGIncGen MLIRSBGAttributesIncGen)
