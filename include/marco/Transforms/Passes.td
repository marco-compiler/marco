#ifndef MARCO_TRANSFORMS_PASSES_TD
#define MARCO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def DataRecomputationPass : Pass<"data-recomputation", "mlir::ModuleOp">
{
  let summary = "Applies data recomputation where likely to be profitable";

  let description = [{
    // TODO
  }];

  let options = [
    Option<"drTestDiagnostics", "dr-test-diagnostics", "bool", "false", "Enables test output">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "vector::VectorDialect", "affine::AffineDialect", "ptr::PtrDialect"
  ];

  let constructor = "mlir::createDataRecomputationPass()";
}

#endif
