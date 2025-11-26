set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Frontend)
add_public_tablegen_target(MLIRFrontendPassesIncGen)
add_mlir_doc(Passes FrontendPasses Dialects/ -gen-pass-doc)
