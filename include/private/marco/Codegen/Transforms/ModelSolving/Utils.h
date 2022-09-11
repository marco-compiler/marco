#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H

#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace marco::codegen
{
  struct ModelSolvingIROptions
  {
    bool mergeAndSortRanges = false;
  };

  void writeDerivativesMap(
      mlir::OpBuilder& builder,
      mlir::modelica::ModelOp modelOp,
      const DerivativesMap& derivativesMap,
      const ModelSolvingIROptions& irOptions);

  mlir::LogicalResult readDerivativesMap(
      mlir::modelica::ModelOp modelOp,
      DerivativesMap& derivativesMap);

  void writeMatchingAttributes(
      mlir::OpBuilder& builder,
      const Model<MatchedEquation>& model,
      const ModelSolvingIROptions& irOptions);

  mlir::LogicalResult readMatchingAttributes(
      const Model<Equation>& model,
      Model<MatchedEquation>& result);

  void writeSchedulingAttributes
      (mlir::OpBuilder& builder,
       const Model<ScheduledEquationsBlock>& model,
       const ModelSolvingIROptions& irOptions);

  mlir::LogicalResult readSchedulingAttributes(
      const Model<MatchedEquation>& model,
      Model<ScheduledEquationsBlock>& result);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
