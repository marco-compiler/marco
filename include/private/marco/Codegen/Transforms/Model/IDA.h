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

      bool hasVariable(mlir::Value variable) const;

      void addVariable(mlir::Value variable);

      void addEquation(ScheduledEquation* equation);
      void processInitFunction(mlir::OpBuilder& builder, Model<ScheduledEquationsBlock> model, mlir::FuncOp initFunction, const mlir::BlockAndValueMapping& derivatives);

    private:
      std::vector<mlir::Value> variables;
      std::set<ScheduledEquation*> equations;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_IDA_H
