#ifndef MARCO_DIALECT_BASEMODELICA_IR_DEFAULTVALUESDEPENDENCYGRAPH_H
#define MARCO_DIALECT_BASEMODELICA_IR_DEFAULTVALUESDEPENDENCYGRAPH_H

#include "marco/Dialect/BaseModelica/IR/VariablesDependencyGraph.h"

namespace mlir::bmodelica
{
  /// Directed graph representing the dependencies among the variables with
  /// respect to the usage of variables for the computation of the default
  /// value.
  class DefaultValuesDependencyGraph : public VariablesDependencyGraph
  {
    public:
      explicit DefaultValuesDependencyGraph(
          const llvm::StringMap<DefaultOp>& defaultOps)
          : defaultOps(&defaultOps)
      {
      }

    protected:
      std::set<llvm::StringRef> getDependencies(VariableOp variable) override;

    private:
      const llvm::StringMap<DefaultOp>* defaultOps;
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_DEFAULTVALUESDEPENDENCYGRAPH_H
