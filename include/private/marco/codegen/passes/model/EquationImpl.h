#ifndef MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H
#define MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H

#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/Path.h"

namespace marco::codegen
{
  class BaseEquation : public Equation
  {
    public:
      BaseEquation(modelica::EquationOp equation, Variables variables);

      modelica::EquationOp getOperation() const override;

      const Variables& getVariables() const override;

      void setVariables(Variables variables) override;

    private:
      mlir::Operation* equationOp;
      Variables variables;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H
