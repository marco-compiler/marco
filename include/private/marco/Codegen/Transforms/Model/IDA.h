#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H

#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include "marco/Codegen/Transforms/Model/ExternalSolver.h"
#include <set>

namespace marco::codegen
{
  class IDASolver : public ExternalSolver
  {
    public:
      IDASolver();

      bool isEnabled() const override;

      mlir::Type getSolverInstanceType(mlir::MLIRContext* context) const override;

      bool hasVariable(mlir::Value variable) const;

      void addVariable(mlir::Value variable);

      bool hasEquation(ScheduledEquation* equation) const;

      void addEquation(ScheduledEquation* equation);

      mlir::LogicalResult init(mlir::OpBuilder& builder, mlir::FuncOp initFunction);

      mlir::LogicalResult processVariables(
          mlir::OpBuilder& builder,
          mlir::FuncOp initFunction,
          const mlir::BlockAndValueMapping& derivatives);

      mlir::LogicalResult processEquations(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          mlir::FuncOp initFunction,
          mlir::TypeRange variableTypes,
          const mlir::BlockAndValueMapping& derivatives);

    private:
      /// The variables of the model that are managed by IDA.
      /// The SSA values are the ones defined by the body of the ModelOp.
      std::vector<mlir::Value> variables;

      std::set<ScheduledEquation*> equations;

      mlir::Value idaInstance;

      /// Map from a ModelOp variable to its IDA variable
      mlir::BlockAndValueMapping mappedVariables;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H
