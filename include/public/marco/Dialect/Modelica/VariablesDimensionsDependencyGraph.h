#ifndef MARCO_DIALECT_MODELICA_VARIABLESDIMENSIONSDEPENDENCYGRAPH_H
#define MARCO_DIALECT_MODELICA_VARIABLESDIMENSIONSDEPENDENCYGRAPH_H

#include "marco/Dialect/Modelica/VariablesDependencyGraph.h"

namespace mlir::modelica
{
  /// Directed graph representing the dependencies among the variables with
  /// respect to the usage of variables for the computation of the dynamic
  /// dimensions of their types.
  class VariablesDimensionsDependencyGraph : public VariablesDependencyGraph
  {
    protected:
      std::set<llvm::StringRef> getDependencies(VariableOp variable) override;
  };
}

#endif // MARCO_DIALECT_MODELICA_VARIABLESDIMENSIONSDEPENDENCYGRAPH_H
