#ifndef MARCO_CODEGEN_PASSES_MODEL_ACCESS_H
#define MARCO_CODEGEN_PASSES_MODEL_ACCESS_H

#include "marco/codegen/passes/model/Path.h"
#include "marco/codegen/passes/model/Variable.h"
#include "marco/modeling/AccessFunction.h"

namespace marco::codegen
{
  class Access
  {
    private:
      using AccessFunction = ::marco::modeling::AccessFunction;

    public:
      Access(Variable* variable, AccessFunction accessFunction, EquationPath path);

      Variable* getVariable() const;

      const AccessFunction& getAccessFunction() const;

      const EquationPath& getPath() const;

    private:
      Variable* variable;
      AccessFunction accessFunction;
      EquationPath path;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_ACCESS_H
