#ifndef MARCO_CODEGEN_TRANSFORMS_SBGMATCHING_H
#define MARCO_CODEGEN_TRANSFORMS_SBGMATCHING_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sbg
{
#define GEN_PASS_DECL_SBGMATCHINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSBGMatchingPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_SBGMATCHING_H