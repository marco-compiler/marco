#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"

using namespace mlir::bmodelica;
using namespace mlir::bmodelica::bridge;

static IndexSet getNonEmptyIndices(IndexSet indices) {
  if (indices.empty()) {
    // Scalar variable.
    indices += MultidimensionalRange(Range(0, 1));
  }

  return std::move(indices);
}

namespace mlir::bmodelica::bridge {
VariableBridge::Id::Id(mlir::SymbolRefAttr name) : name(name) {}

bool VariableBridge::Id::operator<(const VariableBridge::Id &other) const {
  if (auto rootCmp =
          name.getRootReference().compare(other.name.getRootReference());
      rootCmp != 0) {
    return rootCmp < 0;
  }

  size_t l1 = name.getNestedReferences().size();
  size_t l2 = other.name.getNestedReferences().size();

  for (size_t i = 0, e = std::min(l1, l2); i < e; ++i) {
    auto firstNestedRef = name.getNestedReferences()[i].getAttr();
    auto secondNestedRef = other.name.getNestedReferences()[i].getAttr();

    if (auto nestedCmp = firstNestedRef.compare(secondNestedRef);
        nestedCmp != 0) {
      return nestedCmp < 0;
    }
  }

  if (l1 < l2) {
    return true;
  }

  return false;
}

bool VariableBridge::Id::operator==(const Id &other) const {
  return name == other.name;
}

bool VariableBridge::Id::operator!=(const Id &other) const {
  return !(*this == other);
}

std::unique_ptr<VariableBridge> VariableBridge::build(mlir::SymbolRefAttr name,
                                                      IndexSet indices) {
  return std::make_unique<VariableBridge>(name, std::move(indices));
}

std::unique_ptr<VariableBridge> VariableBridge::build(VariableOp variable) {
  auto nameAttr = mlir::SymbolRefAttr::get(variable.getSymNameAttr());
  return build(nameAttr, getNonEmptyIndices(variable.getIndices()));
}

VariableBridge::VariableBridge(mlir::SymbolRefAttr name,
                               marco::modeling::IndexSet indices)
    : id(name), name(name), indices(std::move(indices)) {}
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
VariableTraits<VariableBridge *>::Id
VariableTraits<VariableBridge *>::getId(const Variable *variable) {
  return (*variable)->id;
}

size_t VariableTraits<VariableBridge *>::getRank(const Variable *variable) {
  size_t rank = (*variable)->indices.rank();

  if (rank == 0) {
    return 1;
  }

  return rank;
}

IndexSet
VariableTraits<VariableBridge *>::getIndices(const Variable *variable) {
  const IndexSet &result = (*variable)->indices;

  if (result.empty()) {
    return {Point(0)};
  }

  return result;
}

llvm::raw_ostream &
VariableTraits<VariableBridge *>::dump(const Variable *variable,
                                       llvm::raw_ostream &os) {
  return os << (*variable)->name;
}
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
VariableTraits<VariableBridge *>::Id
VariableTraits<VariableBridge *>::getId(const Variable *variable) {
  return (*variable)->id;
}

size_t VariableTraits<VariableBridge *>::getRank(const Variable *variable) {
  size_t rank = (*variable)->indices.rank();

  if (rank == 0) {
    return 1;
  }

  return rank;
}

IndexSet
VariableTraits<VariableBridge *>::getIndices(const Variable *variable) {
  const IndexSet &result = (*variable)->indices;

  if (result.empty()) {
    return {Point(0)};
  }

  return result;
}

llvm::raw_ostream &
VariableTraits<VariableBridge *>::dump(const Variable *variable,
                                       llvm::raw_ostream &os) {
  return os << (*variable)->name;
}
} // namespace marco::modeling::dependency
