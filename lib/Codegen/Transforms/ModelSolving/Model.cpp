#include "marco/Codegen/Transforms/ModelSolving/Model.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  Variables discoverVariables(mlir::Region& equationsRegion)
  {
    Variables result;

    for (const auto& variable : equationsRegion.getArguments()) {
      result.add(std::make_unique<Variable>(variable));
    }

    return result;
  }

  Equations<Equation> discoverEquations(mlir::Region& equationsRegion, const Variables& variables)
  {
    Equations<Equation> result;

    equationsRegion.walk([&](EquationOp equationOp) {
      result.add(Equation::build(equationOp, variables));
    });

    return result;
  }

  namespace impl
  {
    BaseModel::BaseModel(mlir::modelica::ModelOp modelOp)
        : modelOp(modelOp.getOperation())
    {
    }

    ModelOp BaseModel::getOperation() const
    {
      return mlir::cast<ModelOp>(modelOp);
    }

    Variables BaseModel::getVariables() const
    {
      return variables;
    }

    llvm::ArrayRef<llvm::StringRef> BaseModel::getVariableNames() const
    {
      return variableNames;
    }

    void BaseModel::setVariableNames(llvm::ArrayRef<llvm::StringRef> names)
    {
      variableNames.clear();

      for (const auto& name : names) {
        variableNames.push_back(name);
      }
    }

    void BaseModel::setVariables(Variables value)
    {
      this->variables = std::move(value);
    }

    VariablesMap& BaseModel::getVariablesMap()
    {
      return variablesMap;
    }

    const VariablesMap& BaseModel::getVariablesMap() const
    {
      return variablesMap;
    }

    void BaseModel::setVariablesMap(VariablesMap map)
    {
      variablesMap = std::move(map);
    }

    DerivativesMap& BaseModel::getDerivativesMap()
    {
      return derivativesMap;
    }

    const DerivativesMap& BaseModel::getDerivativesMap() const
    {
      return derivativesMap;
    }

    void BaseModel::setDerivativesMap(DerivativesMap map)
    {
      derivativesMap = std::move(map);
    }
  }
}
