#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_MODEL_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_MODEL_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Variable.h"
#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/Path.h"

namespace marco::codegen
{
  /// Get all the variables that are declared inside the Model operation, independently
  /// from their nature (state variables, constants, etc.).
  Variables discoverVariables(mlir::Region& equationsRegion);

  /// Get the equations that are declared inside the Model operation.
  Equations<Equation> discoverEquations(
      mlir::Region& equationsRegion, const Variables& variables);

  namespace impl
  {
    class BaseModel
    {
      public:
        BaseModel(mlir::modelica::ModelOp modelOp);

        virtual ~BaseModel();

        /// Get the IR model operation
        mlir::modelica::ModelOp getOperation() const;

        /// Get the variables that are managed by this model.
        Variables getVariables() const;

        /// Set the variables the are managed by this model.
        void setVariables(Variables value);

        DerivativesMap& getDerivativesMap();

        const DerivativesMap& getDerivativesMap() const;

        void setDerivativesMap(DerivativesMap map);

      protected:
        /// Hook function that is called when a new set of variables is set.
        virtual void onVariablesSet(Variables newVariables);

      private:
        mlir::Operation* modelOp;
        Variables variables;
        DerivativesMap derivativesMap;
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

        for (auto& equation : this->equations) {
          equation->setVariables(getVariables());
        }
      }

    protected:
      void onVariablesSet(Variables newVariables) override
      {
        for (auto& equation : equations) {
          equation->setVariables(newVariables);
        }
      }

    private:
      Equations<EquationType> equations;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_MODEL_H
