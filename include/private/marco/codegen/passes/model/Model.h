#ifndef MARCO_MATCHING_MODEL_H
#define MARCO_MATCHING_MODEL_H

#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/Path.h"
#include "marco/codegen/passes/model/Variable.h"
#include "mlir/IR/BlockAndValueMapping.h"

namespace marco::codegen
{
  class Model
  {
    public:
      Model(modelica::ModelOp modelOp);

      modelica::ModelOp getOperation() const;

      /// Get the variables that are managed by this model.
      Variables getVariables() const;

      /// Set the variables the are managed by this model.
      void setVariables(Variables variables);

      /// Get the equations that are managed by this model.
      Equations getEquations() const;

      /// Set the equations that are managed by this model.
      void setEquations(Equations equations);

      /*
      mlir::BlockAndValueMapping& getDerivatives();

      const mlir::BlockAndValueMapping& getDerivatives() const;
       */

    private:
      mlir::Operation* modelOp;

      // Map between a value and its derivative
      //mlir::BlockAndValueMapping derivatives;

      Variables variables;
      Equations equations;
  };
}

#endif // MARCO_MATCHING_MODEL_H
