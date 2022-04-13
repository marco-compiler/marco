#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_CYCLES_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_CYCLES_H

#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"

namespace marco::codegen
{
  /// Modify the IR in order to solve the algebraic loops.
  /// Return a success result code if all the cycles have been solved.
  mlir::LogicalResult solveCycles(Model<MatchedEquation>& model, mlir::OpBuilder& builder);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_CYCLES_H
