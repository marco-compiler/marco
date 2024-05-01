#ifndef MARCO_DIALECT_BASEMODELICA_VARIABLEACCESS_H
#define MARCO_DIALECT_BASEMODELICA_VARIABLEACCESS_H

#include "marco/Dialect/BaseModelica/Common.h"
#include "marco/Dialect/BaseModelica/EquationPath.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::bmodelica
{
  using AccessFunction = ::marco::modeling::AccessFunction;

  using AccessFunctionRotoTranslation =
      ::marco::modeling::AccessFunctionRotoTranslation;

  class VariableAccess
  {
    public:
      VariableAccess(
        EquationPath path,
        mlir::SymbolRefAttr variable,
        std::unique_ptr<AccessFunction> accessFunction);

      VariableAccess(const VariableAccess& other);

      ~VariableAccess();

      VariableAccess& operator=(const VariableAccess& other);

      VariableAccess& operator=(VariableAccess&& other);

      friend void swap(VariableAccess& first, VariableAccess& second);

      const EquationPath& getPath() const;

      mlir::SymbolRefAttr getVariable() const;

      const AccessFunction& getAccessFunction() const;

    private:
      EquationPath path;
      mlir::SymbolRefAttr variable;
      std::unique_ptr<AccessFunction> accessFunction;
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_VARIABLEACCESS_H
