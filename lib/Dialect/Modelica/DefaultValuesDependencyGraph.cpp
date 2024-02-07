#include "marco/Dialect/Modelica/DefaultValuesDependencyGraph.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
{
  std::set<llvm::StringRef> DefaultValuesDependencyGraph::getDependencies(
      VariableOp variable)
  {
    std::set<llvm::StringRef> dependencies;
    auto defaultOpIt = defaultOps->find(variable.getSymName());

    if (defaultOpIt != defaultOps->end()) {
      DefaultOp defaultOp = defaultOpIt->getValue();

      defaultOp->walk([&](VariableGetOp getOp) {
        dependencies.insert(getOp.getVariable());
      });
    }

    return dependencies;
  }
}
