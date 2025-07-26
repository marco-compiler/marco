#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"

using namespace mlir::bmodelica;
using namespace mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
VariableBridge::Id::Id(mlir::SymbolRefAttr name) : name(name) {}

bool VariableBridge::Id::operator==(const Id &other) const {
  return name == other.name;
}

bool VariableBridge::Id::operator!=(const Id &other) const {
  return !(*this == other);
}

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

  if (l2 < l1) {
    return false;
  }

  return false;
}

llvm::hash_code hash_value(const VariableBridge::Id &val) {
  return llvm::hash_value(val.name.getRootReference().getValue());
}

namespace {
class VariableBridgeImpl : public VariableBridge {
public:
  template <typename... Arg>
  VariableBridgeImpl(Arg &&...arg)
      : VariableBridge(std::forward<Arg>(arg)...) {}
};
} // namespace

VariableBridge &Storage::addVariable(mlir::SymbolRefAttr name,
                                     IndexSet indices) {
  auto &ptr = variableBridges.emplace_back(
      std::make_unique<VariableBridgeImpl>(name, std::move(indices)));

  variablesMap[ptr->getName()] = ptr.get();
  return *ptr;
}

VariableBridge &Storage::addVariable(VariableOp variable) {
  auto nameAttr = mlir::SymbolRefAttr::get(variable.getSymNameAttr());
  return addVariable(nameAttr, variable.getIndices());
}

VariableBridge::VariableBridge(mlir::SymbolRefAttr name, IndexSet indices)
    : id(name), name(name), indices(std::move(indices)) {}

llvm::hash_code hash_value(const VariableBridge &val) {
  return hash_value(val.id);
}

size_t VariableBridge::getOriginalRank() const {
  return getOriginalIndices().rank();
}

const IndexSet &VariableBridge::getOriginalIndices() const { return indices; }

IndexSet VariableBridge::getIndices() const {
  if (const IndexSet &originalIndices = getOriginalIndices();
      !originalIndices.empty()) {
    return originalIndices;
  }

  return {Point(0)};
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const VariableBridge::Id &obj) {
  return os << obj.name;
}
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
VariableTraits<VariableBridge *>::Id
VariableTraits<VariableBridge *>::getId(const Variable *variable) {
  return (*variable)->getId();
}

size_t VariableTraits<VariableBridge *>::getRank(const Variable *variable) {
  return (*variable)->getIndices().rank();
}

IndexSet
VariableTraits<VariableBridge *>::getIndices(const Variable *variable) {
  return (*variable)->getIndices();
}

llvm::raw_ostream &
VariableTraits<VariableBridge *>::dump(const Variable *variable,
                                       llvm::raw_ostream &os) {
  return os << (*variable)->getId();
}
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
VariableTraits<VariableBridge *>::Id
VariableTraits<VariableBridge *>::getId(const Variable *variable) {
  return (*variable)->getId();
}

size_t VariableTraits<VariableBridge *>::getRank(const Variable *variable) {
  return (*variable)->getIndices().rank();
}

IndexSet
VariableTraits<VariableBridge *>::getIndices(const Variable *variable) {
  return (*variable)->getIndices();
}

llvm::raw_ostream &
VariableTraits<VariableBridge *>::dump(const Variable *variable,
                                       llvm::raw_ostream &os) {
  return os << (*variable)->getId();
}
} // namespace marco::modeling::dependency
