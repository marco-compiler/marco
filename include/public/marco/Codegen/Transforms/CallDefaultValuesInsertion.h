#ifndef MARCO_CODEGEN_TRANSFORMS_CALLDEFAULTVALUESINSERTION_H
#define MARCO_CODEGEN_TRANSFORMS_CALLDEFAULTVALUESINSERTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_CALLDEFAULTVALUESINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createCallDefaultValuesInsertionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_CALLDEFAULTVALUESINSERTION_H
