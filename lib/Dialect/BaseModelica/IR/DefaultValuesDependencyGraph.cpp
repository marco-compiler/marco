#include "marco/Dialect/BaseModelica/IR/DefaultValuesDependencyGraph.h"
#include "llvm/ADT/StringSet.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
llvm::SmallVector<std::string>
DefaultValuesDependencyGraph::getDependencies(VariableOp variable) {
  llvm::SmallVector<std::string> dependencies;
  llvm::StringSet<> uniqueNames;
  auto defaultOpIt = defaultOps->find(variable.getSymName());

  if (defaultOpIt != defaultOps->end()) {
    DefaultOp defaultOp = defaultOpIt->getValue();

    defaultOp->walk([&](VariableGetOp getOp) {
      if (!uniqueNames.contains(getOp.getVariable())) {
        uniqueNames.insert(getOp.getVariable());
        dependencies.push_back(getOp.getVariable().str());
      }
    });
  }

  llvm::sort(dependencies);
  return dependencies;
}
} // namespace mlir::bmodelica
