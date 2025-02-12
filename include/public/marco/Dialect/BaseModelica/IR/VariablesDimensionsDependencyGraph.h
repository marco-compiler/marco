#ifndef MARCO_DIALECT_BASEMODELICA_IR_VARIABLESDIMENSIONSDEPENDENCYGRAPH_H
#define MARCO_DIALECT_BASEMODELICA_IR_VARIABLESDIMENSIONSDEPENDENCYGRAPH_H

#include "marco/Dialect/BaseModelica/IR/VariablesDependencyGraph.h"

namespace mlir::bmodelica {
/// Directed graph representing the dependencies among the variables with
/// respect to the usage of variables for the computation of the dynamic
/// dimensions of their types.
class VariablesDimensionsDependencyGraph : public VariablesDependencyGraph {
protected:
  std::set<llvm::StringRef> getDependencies(VariableOp variable) override;
};
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_IR_VARIABLESDIMENSIONSDEPENDENCYGRAPH_H
