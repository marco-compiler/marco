#ifndef MARCO_CODEGEN_TRANSFORMS_ACCESSREPLACEMENTTEST_H
#define MARCO_CODEGEN_TRANSFORMS_ACCESSREPLACEMENTTEST_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_ACCESSREPLACEMENTTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createAccessReplacementTestPass();
}


#endif // MARCO_CODEGEN_TRANSFORMS_ACCESSREPLACEMENTTEST_H
