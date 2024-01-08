#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONSIDESSPLIT_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONSIDESSPLIT_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_EQUATIONSIDESSPLITPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationSidesSplitPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONSIDESSPLIT_H
