#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
std::unique_ptr<AccessFunction>
convertAccessFunction(mlir::MLIRContext *context,
                      const mlir::bmodelica::VariableAccess &access) {
  const AccessFunction &accessFunction = access.getAccessFunction();

  if (accessFunction.getNumOfResults() == 0) {
    // Access to scalar variable.
    return AccessFunction::build(mlir::AffineMap::get(
        std::max(accessFunction.getNumOfDims(), static_cast<uint64_t>(1)), 0,
        mlir::getAffineConstantExpr(0, context)));
  }

  if (accessFunction.getNumOfDims() == 0) {
    return accessFunction.getWithGivenDimensions(1);
  }

  return accessFunction.clone();
}

class EquationBridgeImpl : public EquationBridge {
public:
  template <typename... Arg>
  EquationBridgeImpl(Arg &&...arg)
      : EquationBridge(std::forward<Arg>(arg)...) {}
};
} // namespace

namespace mlir::bmodelica::bridge {
EquationBridge::AccessesList::AccessesList() = default;

EquationBridge::AccessesList::AccessesList(Reference accesses)
    : accesses(std::move(accesses)) {}

EquationBridge::AccessesList::AccessesList(Container accesses)
    : accesses(std::move(accesses)) {}

EquationBridge::AccessesList::operator llvm::ArrayRef<VariableAccess>() const {
  if (std::holds_alternative<Reference>(accesses)) {
    return std::get<Reference>(accesses);
  }

  return std::get<Container>(accesses);
}

EquationBridge &
Storage::addEquation(uint64_t id, EquationInstanceOp op,
                     mlir::SymbolTableCollection &symbolTableCollection) {
  auto &ptr = equationBridges.emplace_back(std::make_unique<EquationBridgeImpl>(
      id, op, symbolTableCollection, *this));

  equationsMap[ptr->getId()] = ptr.get();
  return *ptr;
}

EquationBridge::EquationBridge(uint64_t id, EquationInstanceOp op,
                               mlir::SymbolTableCollection &symbolTable,
                               const Storage &storage)
    : id(id), op(op), storage(storage), symbolTable(&symbolTable),
      accessAnalysis(nullptr) {}

llvm::hash_code hash_value(const EquationBridge &val) {
  return llvm::hash_value(val.id);
}

EquationBridge::Id EquationBridge::getId() const { return id; }

EquationInstanceOp EquationBridge::getOp() const { return op; }

mlir::SymbolTableCollection &EquationBridge::getSymbolTableCollection() {
  assert(symbolTable);
  return *symbolTable;
}

const mlir::SymbolTableCollection &
EquationBridge::getSymbolTableCollection() const {
  assert(symbolTable);
  return *symbolTable;
}

bool EquationBridge::hasAccessAnalysis() const {
  return accessAnalysis != nullptr;
}

VariableAccessAnalysis &EquationBridge::getAccessAnalysis() {
  assert(hasAccessAnalysis());
  return *accessAnalysis;
}

const VariableAccessAnalysis &EquationBridge::getAccessAnalysis() const {
  assert(hasAccessAnalysis());
  return *accessAnalysis;
}

void EquationBridge::setAccessAnalysis(VariableAccessAnalysis &accessAnalysis) {
  this->accessAnalysis = &accessAnalysis;
}

size_t EquationBridge::getOriginalRank() const {
  return getOp().getInductionVariables().size();
}

IndexSet EquationBridge::getOriginalIndices() const {
  return getOp().getIterationSpace();
}

EquationBridge::AccessesList EquationBridge::getOriginalAccesses() {
  if (hasAccessAnalysis()) {
    if (auto cachedAccesses =
            getAccessAnalysis().getAccesses(getSymbolTableCollection())) {
      return AccessesList(*cachedAccesses);
    }
  }

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::succeeded(
          getOp().getAccesses(accesses, getSymbolTableCollection()))) {
    return AccessesList(std::move(accesses));
  }

  llvm_unreachable("Can't compute the accesses");
  return {};
}

EquationBridge::AccessesList EquationBridge::getOriginalWriteAccesses() {
  AccessesList accesses = getOriginalAccesses();
  llvm::SmallVector<VariableAccess> writeAccesses;

  if (mlir::failed(getOp().getWriteAccesses(writeAccesses,
                                            getSymbolTableCollection(),
                                            getOriginalIndices(), accesses))) {
    llvm_unreachable("Can't compute write accesses");
    return {};
  }

  return AccessesList(std::move(writeAccesses));
}

EquationBridge::AccessesList EquationBridge::getOriginalReadAccesses() {
  AccessesList accesses = getOriginalAccesses();
  llvm::SmallVector<VariableAccess> readAccesses;

  if (mlir::failed(getOp().getReadAccesses(readAccesses,
                                           getSymbolTableCollection(),
                                           getOriginalIndices(), accesses))) {
    llvm_unreachable("Can't compute write accesses");
    return {};
  }

  return AccessesList(std::move(readAccesses));
}

size_t EquationBridge::getRank() const {
  if (auto rank = getOriginalRank(); rank != 0) {
    return rank;
  }

  return 1;
}

IndexSet EquationBridge::getIndices() const {
  if (auto originalIndices = getOriginalIndices(); !originalIndices.empty()) {
    return originalIndices;
  }

  return {Point(0)};
}

void EquationBridge::walkAccesses(AccessWalkFn callbackFn) {
  for (const VariableAccess &access : getOriginalAccesses()) {
    auto accessFunction = convertAccessFunction(getOp().getContext(), access);
    mlir::SymbolRefAttr variable = access.getVariable();

    if (storage.hasVariable(variable)) {
      callbackFn(access, &storage.getVariable(variable), *accessFunction);
    }
  }
}

void EquationBridge::walkWriteAccesses(AccessWalkFn callbackFn) {
  for (const VariableAccess &access : getOriginalWriteAccesses()) {
    auto accessFunction = convertAccessFunction(getOp().getContext(), access);
    mlir::SymbolRefAttr variable = access.getVariable();

    if (storage.hasVariable(variable)) {
      callbackFn(access, &storage.getVariable(variable), *accessFunction);
    }
  }
}

void EquationBridge::walkReadAccesses(AccessWalkFn callbackFn) {
  for (const VariableAccess &access : getOriginalReadAccesses()) {
    auto accessFunction = convertAccessFunction(getOp().getContext(), access);
    mlir::SymbolRefAttr variable = access.getVariable();

    if (storage.hasVariable(variable)) {
      callbackFn(access, &storage.getVariable(variable), *accessFunction);
    }
  }
}

llvm::hash_code hash_value(const EquationBridge *val) {
  return hash_value(*val);
}
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
EquationTraits<EquationBridge *>::Id
EquationTraits<EquationBridge *>::getId(const Equation *equation) {
  return (*equation)->getId();
}

size_t EquationTraits<EquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  return (*equation)->getRank();
}

IndexSet
EquationTraits<EquationBridge *>::getIndices(const Equation *equation) {
  return (*equation)->getIndices();
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableId>>
EquationTraits<EquationBridge *>::getAccesses(const Equation *equation) {
  std::vector<Access<EquationTraits<EquationBridge *>::VariableId>> result;

  (*equation)->walkAccesses([&](const VariableAccess &access,
                                VariableBridge *variable,
                                const AccessFunction &accessFunction) {
    result.push_back(convertAccess(access, variable, accessFunction));
  });

  return result;
}

llvm::raw_ostream &
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->getOp().printInline(os);
  return os;
}

Access<EquationTraits<EquationBridge *>::VariableId>
EquationTraits<EquationBridge *>::convertAccess(
    const VariableAccess &access, VariableBridge *variable,
    const AccessFunction &accessFunction) {
  return Access<EquationTraits<EquationBridge *>::VariableId>(
      variable->getId(), accessFunction.clone());
}
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
EquationTraits<EquationBridge *>::Id
EquationTraits<EquationBridge *>::getId(const Equation *equation) {
  return (*equation)->getId();
}

size_t EquationTraits<EquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  return (*equation)->getRank();
}

IndexSet
EquationTraits<EquationBridge *>::getIterationRanges(const Equation *equation) {
  return (*equation)->getIndices();
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getAccesses(const Equation *equation) {
  std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                     EquationTraits<EquationBridge *>::AccessProperty>>
      result;

  (*equation)->walkAccesses([&](const VariableAccess &access,
                                VariableBridge *variable,
                                const AccessFunction &accessFunction) {
    result.push_back(convertAccess(access, variable, accessFunction));
  });

  return result;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getWrites(const Equation *equation) {
  std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                     EquationTraits<EquationBridge *>::AccessProperty>>
      result;

  (*equation)->walkWriteAccesses([&](const VariableAccess &access,
                                     VariableBridge *variable,
                                     const AccessFunction &accessFunction) {
    result.push_back(convertAccess(access, variable, accessFunction));
  });

  return result;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getReads(const Equation *equation) {
  std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                     EquationTraits<EquationBridge *>::AccessProperty>>
      result;

  (*equation)->walkReadAccesses([&](const VariableAccess &access,
                                    VariableBridge *variable,
                                    const AccessFunction &accessFunction) {
    result.push_back(convertAccess(access, variable, accessFunction));
  });

  return result;
}

llvm::raw_ostream &
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->getOp().printInline(os);
  return os;
}

Access<EquationTraits<EquationBridge *>::VariableType,
       EquationTraits<EquationBridge *>::AccessProperty>
EquationTraits<EquationBridge *>::convertAccess(
    const VariableAccess &access, VariableBridge *variable,
    const AccessFunction &accessFunction) {

  return Access<EquationTraits<EquationBridge *>::VariableType,
                EquationTraits<EquationBridge *>::AccessProperty>(
      variable, accessFunction.clone(), access);
}
} // namespace marco::modeling::dependency
