#ifndef MARCO_CODEGEN_CONVERSION_MODELICATOSBG_MODELICATOSBG_H
#define MARCO_CODEGEN_CONVERSION_MODELICATOSBG_MODELICATOSBG_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DECL_MODELICATOSBGINSERTIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createModelicaToSBGInsertionPass();
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICATOSBG_MODELICATOSBG_H
