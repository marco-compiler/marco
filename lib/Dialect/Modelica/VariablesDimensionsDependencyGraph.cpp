#include "marco/Dialect/Modelica/VariablesDimensionsDependencyGraph.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
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
