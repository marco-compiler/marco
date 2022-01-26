#include "marco/codegen/passes/model/Access.h"

using namespace ::marco::modeling;

namespace marco::codegen
{
  Access::Access(Variable* variable, AccessFunction accessFunction, EquationPath path)
    : variable(std::move(variable)),
      accessFunction(std::move(accessFunction)),
      path(std::move(path))
  {
  }

  Variable* Access::getVariable() const
  {
    return variable;
  }

  const AccessFunction& Access::getAccessFunction() const
  {
    return accessFunction;
  }

  const EquationPath& Access::getPath() const
  {
    return path;
  }
}
