#ifndef MARCO_CODEGEN_TRANSFORMS_EQUATIONEXPLICITATIONTEST_H
#define MARCO_CODEGEN_TRANSFORMS_EQUATIONEXPLICITATIONTEST_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::modelica
{
#define GEN_PASS_DECL_EQUATIONEXPLICITATIONTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationExplicitationTestPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_EQUATIONEXPLICITATIONTEST_H
