#ifndef MARCO_CONSTANTFOLDING_H
#define MARCO_CONSTANTFOLDING_H

#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_CONSTANTFOLDINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
  std::unique_ptr<mlir::Pass> createConstantFoldingPass();

  std::unique_ptr<mlir::Pass> createConstantFoldingPass(const ConstantFoldingPassOptions& options);
}

#endif//MARCO_CONSTANTFOLDING_H
