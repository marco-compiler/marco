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

    void BaseModel::setVariables(Variables value)
    {
      this->variables = std::move(value);
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
