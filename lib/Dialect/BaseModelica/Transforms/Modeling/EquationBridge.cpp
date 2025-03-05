#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
std::unique_ptr<AccessFunction>
convertAccessFunction(mlir::MLIRContext *context,
                      const mlir::bmodelica::VariableAccess &access) {
  const AccessFunction &accessFunction = access.getAccessFunction();

  if (accessFunction.getNumOfResults() == 0) {
    // Access to scalar variable.
    return AccessFunction::build(
        mlir::AffineMap::get(accessFunction.getNumOfDims(), 0,
                             mlir::getAffineConstantExpr(0, context)));
  }

  return accessFunction.clone();
}
} // namespace

namespace mlir::bmodelica::bridge {
EquationBridge::EquationBridge(
    int64_t id, EquationInstanceOp op, mlir::SymbolTableCollection &symbolTable,
    llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap)
    : id(id), op(op), symbolTable(&symbolTable), accessAnalysis(nullptr),
      variablesMap(&variablesMap) {}

EquationBridge::EquationBridge(
    int64_t id, EquationInstanceOp op, mlir::SymbolTableCollection &symbolTable,
    VariableAccessAnalysis &accessAnalysis,
    llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap)
    : id(id), op(op), symbolTable(&symbolTable),
      accessAnalysis(&accessAnalysis), variablesMap(&variablesMap) {}

int64_t EquationBridge::getId() const { return id; }

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

EquationBridge::VariablesMap &EquationBridge::getVariablesMap() {
  assert(variablesMap);
  return *variablesMap;
}

const EquationBridge::VariablesMap &EquationBridge::getVariablesMap() const {
  assert(variablesMap);
  return *variablesMap;
}
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
EquationTraits<EquationBridge *>::Id
EquationTraits<EquationBridge *>::getId(const Equation *equation) {
  return (*equation)->getId();
}

size_t EquationTraits<EquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  auto numOfInductions = static_cast<uint64_t>(
      (*equation)->getOp().getInductionVariables().size());

  if (numOfInductions == 0) {
    // Scalar equation.
    return 1;
  }

  return static_cast<size_t>(numOfInductions);
}

IndexSet
EquationTraits<EquationBridge *>::getIterationRanges(const Equation *equation) {
  IndexSet iterationSpace = (*equation)->getOp().getProperties().indices;

  if (iterationSpace.empty()) {
    // Scalar equation.
    iterationSpace += MultidimensionalRange(Range(0, 1));
  }

  return iterationSpace;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getAccesses(const Equation *equation) {
  if ((*equation)->hasAccessAnalysis()) {
    if (auto cachedAccesses = (*equation)->getAccessAnalysis().getAccesses(
            (*equation)->getSymbolTableCollection())) {
      return convertAccesses(equation, *cachedAccesses);
    }
  }

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::succeeded((*equation)->getOp().getAccesses(
          accesses, (*equation)->getSymbolTableCollection()))) {
    return convertAccesses(equation, accesses);
  }

  llvm_unreachable("Can't compute the accesses");
  return {};
}

llvm::raw_ostream &
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->getOp().printInline(os);
  return os;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::convertAccesses(
    const Equation *equation,
    llvm::ArrayRef<mlir::bmodelica::VariableAccess> accesses) {
  std::vector<Access<VariableType, AccessProperty>> result;

  for (const auto &access : accesses) {
    auto accessFunction =
        convertAccessFunction((*equation)->getOp().getContext(), access);

    auto variableIt = (*equation)->getVariablesMap().find(access.getVariable());

    if (variableIt != (*equation)->getVariablesMap().end()) {
      result.emplace_back(variableIt->getSecond(), std::move(accessFunction),
                          access.getPath());
    }
  }

  return result;
}
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
EquationTraits<EquationBridge *>::Id
EquationTraits<EquationBridge *>::getId(const Equation *equation) {
  return (*equation)->getId();
}

size_t EquationTraits<EquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  uint64_t numOfInductions = static_cast<uint64_t>(
      (*equation)->getOp().getInductionVariables().size());

  if (numOfInductions == 0) {
    // Scalar equation.
    return 1;
  }

  return static_cast<size_t>(numOfInductions);
}

IndexSet
EquationTraits<EquationBridge *>::getIterationRanges(const Equation *equation) {
  IndexSet iterationSpace = (*equation)->getOp().getIterationSpace();

  if (iterationSpace.empty()) {
    // Scalar equation.
    iterationSpace += MultidimensionalRange(Range(0, 1));
  }

  return iterationSpace;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getAccesses(const Equation *equation) {
  std::vector<Access<VariableType, AccessProperty>> result;
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed((*equation)->getOp().getAccesses(
          accesses, (*equation)->getSymbolTableCollection()))) {
    return result;
  }

  for (VariableAccess &access : accesses) {
    auto accessFunction =
        convertAccessFunction((*equation)->getOp().getContext(), access);

    auto variableIt = (*equation)->getVariablesMap().find(access.getVariable());

    if (variableIt != (*equation)->getVariablesMap().end()) {
      result.emplace_back(variableIt->getSecond(), std::move(accessFunction),
                          access);
    }
  }

  return result;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getWrites(const Equation *equation) {
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed((*equation)->getOp().getAccesses(
          accesses, (*equation)->getSymbolTableCollection()))) {
    llvm_unreachable("Can't compute the accesses");
    return {};
  }

  llvm::SmallVector<VariableAccess> writeAccesses;

  if (mlir::failed((*equation)->getOp().getWriteAccesses(
          writeAccesses, (*equation)->getSymbolTableCollection(),
          (*equation)->getOp().getProperties().indices, accesses))) {
    llvm_unreachable("Can't compute write accesses");
    return {};
  }

  std::vector<Access<VariableType, AccessProperty>> writes;

  for (const VariableAccess &access : writeAccesses) {
    auto variableIt = (*equation)->getVariablesMap().find(access.getVariable());

    writes.emplace_back(
        variableIt->getSecond(),
        convertAccessFunction((*equation)->getOp().getContext(), access),
        access);
  }

  return writes;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getReads(const Equation *equation) {
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed((*equation)->getOp().getAccesses(
          accesses, (*equation)->getSymbolTableCollection()))) {
    llvm_unreachable("Can't compute the accesses");
    return {};
  }

  llvm::SmallVector<VariableAccess> readAccesses;

  if (mlir::failed((*equation)->getOp().getReadAccesses(
          readAccesses, (*equation)->getSymbolTableCollection(),
          (*equation)->getOp().getProperties().indices, accesses))) {
    llvm_unreachable("Can't compute read accesses");
    return {};
  }

  std::vector<Access<VariableType, AccessProperty>> reads;

  for (const VariableAccess &access : readAccesses) {
    auto variableIt = (*equation)->getVariablesMap().find(access.getVariable());

    reads.emplace_back(
        variableIt->getSecond(),
        convertAccessFunction((*equation)->getOp().getContext(), access),
        access);
  }

  return reads;
}

llvm::raw_ostream &
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->getOp().printInline(os);
  return os;
}
} // namespace marco::modeling::dependency
