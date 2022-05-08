#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLEIMPL_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLEIMPL_H

#include "marco/Codegen/Transforms/ModelSolving/Variable.h"

namespace marco::codegen
{
  class BaseVariable : public Variable
  {
    public:
      using Id = Variable::Id;

      BaseVariable(mlir::Value value);

      Id getId() const override;

      mlir::Value getValue() const override;
      mlir::modelica::MemberCreateOp getDefiningOp() const override;
      bool isConstant() const override;

    private:
      mlir::Value value;
      mlir::Operation* definingOp;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLEIMPL_H
