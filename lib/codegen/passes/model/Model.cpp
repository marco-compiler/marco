#include "marco/codegen/passes/model/Model.h"

using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  Model::Model(ModelOp modelOp) : modelOp(modelOp.getOperation())
  {
  }

  ModelOp Model::getOperation() const
  {
    return mlir::cast<ModelOp>(modelOp);
  }

  Variables Model::getVariables() const
  {
    return variables;
  }

  void Model::setVariables(Variables value)
  {
    this->variables = value;
  }

  Equations Model::getEquations() const
  {
    return equations;
  }

  void Model::setEquations(Equations value)
  {
    this->equations = value;
  }

  /*
  mlir::BlockAndValueMapping& Model::getDerivatives()
  {
    return derivatives;
  }

  const mlir::BlockAndValueMapping& Model::getDerivatives() const
  {
    return derivatives;
  }
   */
}
