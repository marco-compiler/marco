#include "marco/Dialect/BaseModelica/IR/DefaultValuesDependencyGraph.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
std::set<llvm::StringRef>
DefaultValuesDependencyGraph::getDependencies(VariableOp variable) {
  std::set<llvm::StringRef> dependencies;
  auto defaultOpIt = defaultOps->find(variable.getSymName());

  if (defaultOpIt != defaultOps->end()) {
    DefaultOp defaultOp = defaultOpIt->getValue();

    defaultOp->walk(
        [&](VariableGetOp getOp) { dependencies.insert(getOp.getVariable()); });
  }

  return dependencies;
}
} // namespace mlir::bmodelica
