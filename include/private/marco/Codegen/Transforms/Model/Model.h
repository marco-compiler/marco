#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_MODEL_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_MODEL_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/Model/Equation.h"
#include "marco/Codegen/Transforms/Model/Path.h"
#include "marco/Codegen/Transforms/Model/Variable.h"

namespace marco::codegen
{
  namespace impl
  {
    class BaseModel
    {
      public:
        BaseModel(mlir::modelica::ModelOp modelOp)
          : modelOp(modelOp.getOperation())
        {
        }

        mlir::modelica::ModelOp getOperation() const
        {
          return mlir::cast<mlir::modelica::ModelOp>(modelOp);
        }

        /// Get the variables that are managed by this model.
        Variables getVariables() const
        {
          return variables;
        }

        /// Set the variables the are managed by this model.
        void setVariables(Variables value)
        {
          this->variables = std::move(value);
        }

      private:
        mlir::Operation* modelOp;
        Variables variables;
    };
  }

  template<typename EquationType = Equation>
  class Model : public impl::BaseModel
  {
    public:
      Model(mlir::modelica::ModelOp modelOp)
        : impl::BaseModel(std::move(modelOp))
      {
      }

      /// Get the equations that are managed by this model.
      Equations<EquationType> getEquations() const
      {
        return equations;
      }

      /// Set the equations that are managed by this model.
      void setEquations(Equations<EquationType> value)
      {
        this->equations = std::move(value);
      }

    private:
      Equations<EquationType> equations;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_MODEL_H
