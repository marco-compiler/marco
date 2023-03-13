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
    bool singleMatchAttr = false;
    bool singleScheduleAttr = false;
  };

  void writeDerivativesMap(
      mlir::OpBuilder& builder,
      mlir::modelica::ModelOp modelOp,
      const mlir::SymbolTable& symbolTable,
      const DerivativesMap& derivativesMap,
      const ModelSolvingIROptions& irOptions);

  mlir::LogicalResult readDerivativesMap(
      mlir::modelica::ModelOp modelOp,
      DerivativesMap& derivativesMap);

  void writeMatchingAttributes(
      mlir::OpBuilder& builder,
      Model<MatchedEquation>& model,
      const ModelSolvingIROptions& irOptions);

  mlir::LogicalResult readMatchingAttributes(
      Model<MatchedEquation>& result,
      std::function<bool(mlir::modelica::EquationInterface)> equationsFilter);

  void writeSchedulingAttributes
      (mlir::OpBuilder& builder,
       Model<ScheduledEquationsBlock>& model,
       const ModelSolvingIROptions& irOptions);

  mlir::LogicalResult readSchedulingAttributes(
      Model<ScheduledEquationsBlock>& result,
      std::function<bool(mlir::modelica::EquationInterface)> equationsFilter);
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_UTILS_H
