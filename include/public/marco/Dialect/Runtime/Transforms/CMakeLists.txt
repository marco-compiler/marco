set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Runtime)
add_public_tablegen_target(MLIRRuntimeTransformsIncGen)

add_mlir_doc(Passes RuntimePasses ./ -gen-pass-doc)
