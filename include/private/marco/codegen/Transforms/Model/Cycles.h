#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_CYCLES_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_CYCLES_H

#include "marco/Codegen/Transforms/Model/Model.h"
#include "marco/Codegen/Transforms/Model/Matching.h"

namespace marco::codegen
{
  /// Modify the IR in order to solve the algebraic loops
  mlir::LogicalResult solveAlgebraicLoops(Model<MatchedEquation>& model, mlir::OpBuilder& builder);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_CYCLES_H
