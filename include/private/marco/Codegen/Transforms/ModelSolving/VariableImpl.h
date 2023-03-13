#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLEIMPL_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLEIMPL_H

#include "marco/Codegen/Transforms/ModelSolving/Variable.h"

namespace marco::codegen
{
  class BaseVariable : public Variable
  {
    public:
      using Id = Variable::Id;

      BaseVariable(mlir::modelica::VariableOp memberCreateOp);

      Id getId() const override;

      mlir::modelica::VariableOp getDefiningOp() const override;

      bool isReadOnly() const override;
      bool isParameter() const override;
      bool isConstant() const override;

    private:
      mlir::Operation* definingOp;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLEIMPL_H
