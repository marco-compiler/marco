#include "marco/Dialect/BaseModelica/VariablesDimensionsDependencyGraph.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica
{
  std::set<llvm::StringRef>
  VariablesDimensionsDependencyGraph::getDependencies(VariableOp variable)
  {
    std::set<llvm::StringRef> dependencies;
    mlir::Region& region = variable.getConstraintsRegion();

    for (VariableGetOp user : region.getOps<VariableGetOp>()) {
      dependencies.insert(user.getVariable());
    }

    return dependencies;
  }
}
