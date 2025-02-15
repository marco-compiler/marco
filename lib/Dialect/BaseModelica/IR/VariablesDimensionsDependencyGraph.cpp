#include "marco/Dialect/BaseModelica/IR/VariablesDimensionsDependencyGraph.h"
#include "llvm/ADT/StringSet.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
llvm::SmallVector<std::string>
VariablesDimensionsDependencyGraph::getDependencies(VariableOp variable) {
  llvm::SmallVector<std::string> dependencies;
  llvm::StringSet<> uniqueNames;

  mlir::Region &region = variable.getConstraintsRegion();

  for (VariableGetOp user : region.getOps<VariableGetOp>()) {
    if (!uniqueNames.contains(user.getVariable())) {
      uniqueNames.insert(user.getVariable());
      dependencies.push_back(user.getVariable().str());
    }
  }

  llvm::sort(dependencies);
  return dependencies;
}
} // namespace mlir::bmodelica
