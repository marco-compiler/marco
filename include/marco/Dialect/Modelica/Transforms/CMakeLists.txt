set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Modelica)
add_public_tablegen_target(MLIRModelicaTransformsIncGen)

add_mlir_doc(Passes ModelicaPasses Dialects/ -gen-pass-doc)
