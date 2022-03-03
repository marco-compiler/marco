#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_ACCESS_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_ACCESS_H

#include "marco/Codegen/Transforms/Model/Path.h"
#include "marco/Codegen/Transforms/Model/Variable.h"
#include "marco/Modeling/AccessFunction.h"

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

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_ACCESS_H
