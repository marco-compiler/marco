set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CustomPasses)
add_public_tablegen_target(MLIRCustomTransformsPassIncGen)

# add_mlir_doc(Passes BaseModelicaPasses Dialects/ -gen-pass-doc)
