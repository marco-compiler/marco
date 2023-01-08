#ifndef MARCO_CODEGEN_TRANSFORMS_CYCLESSOLVING_H
#define MARCO_CODEGEN_TRANSFORMS_CYCLESSOLVING_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_CYCLESSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createCyclesSolvingPass();

  std::unique_ptr<mlir::Pass> createCyclesSolvingPass(
      const CyclesSolvingPassOptions& options);
}

#endif // MARCO_CODEGEN_TRANSFORMS_CYCLESSOLVING_H
