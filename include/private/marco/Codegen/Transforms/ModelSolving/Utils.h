#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H

#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace marco::codegen
{
  DerivativesMap getDerivativesMap(mlir::modelica::ModelOp modelOp);

  void setDerivativesMap(mlir::modelica::ModelOp modelOp, const DerivativesMap& derivativesMap);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
