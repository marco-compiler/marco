#ifndef MARCO_CODEGEN_TRANSFORMS_SCHEDULERSINSTANTIATION_H
#define MARCO_CODEGEN_TRANSFORMS_SCHEDULERSINSTANTIATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_SCHEDULERSINSTANTIATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createSchedulersInstantiationPass();
}


#endif // MARCO_CODEGEN_TRANSFORMS_SCHEDULERSINSTANTIATION_H
