#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H

#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace marco::codegen
{
  DerivativesMap getDerivativesMap(mlir::modelica::ModelOp modelOp);

  void setDerivativesMap(mlir::OpBuilder& builder, mlir::modelica::ModelOp modelOp, const DerivativesMap& derivativesMap);

  void writeMatchingAttributes(mlir::OpBuilder& builder, const Model<MatchedEquation>& model);

  void readMatchingAttributes(const Model<Equation>& model, Model<MatchedEquation>& result);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
