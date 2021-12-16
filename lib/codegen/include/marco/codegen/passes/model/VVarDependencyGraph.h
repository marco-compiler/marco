#ifndef MARCO_VVARDEPENDENCYGRAPH_H
#define MARCO_VVARDEPENDENCYGRAPH_H

#include <marco/codegen/dialects/modelica/ModelicaDialect.h>

namespace marco::codegen
{
  class VVarDependencyGraph
  {
    public:
    void add(modelica::EquationOp equation);

    private:

  };
}

#endif //MARCO_VVARDEPENDENCYGRAPH_H
