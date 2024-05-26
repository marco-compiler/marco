#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_PRINTMODELINFO_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_PRINTMODELINFO_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_PRINTMODELINFOPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createPrintModelInfoPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_PRINTMODELINFO_H
