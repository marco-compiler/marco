set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BaseModelica)
add_public_tablegen_target(MLIRBaseModelicaTransformsIncGen)

add_mlir_doc(Passes BaseModelicaPasses Dialects/ -gen-pass-doc)
