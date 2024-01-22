#include "marco/Codegen/Transforms/Modeling/VariableBridge.h"

using namespace mlir::modelica;
using namespace mlir::modelica::bridge;

static IndexSet getNonEmptyIndices(IndexSet indices)
{
  if (indices.empty()) {
    // Scalar variable.
    indices += MultidimensionalRange(Range(0, 1));
  }

  return std::move(indices);
}

namespace mlir::modelica::bridge
{
  std::unique_ptr<VariableBridge> VariableBridge::build(
      mlir::SymbolRefAttr name,
      IndexSet indices)
  {
    return std::make_unique<VariableBridge>(name, std::move(indices));
  }

  std::unique_ptr<VariableBridge> VariableBridge::build(VariableOp variable)
  {
    auto nameAttr = mlir::SymbolRefAttr::get(variable.getSymNameAttr());
    return build(nameAttr, getNonEmptyIndices(variable.getIndices()));
  }

  VariableBridge::VariableBridge(
      mlir::SymbolRefAttr name,
      marco::modeling::IndexSet indices)
      : name(name),
        indices(std::move(indices))
  {
  }
}