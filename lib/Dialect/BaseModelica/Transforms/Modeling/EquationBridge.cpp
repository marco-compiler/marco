#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
std::unique_ptr<AccessFunction>
getAccessFunction(mlir::MLIRContext *context,
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
    VariableAccessAnalysis &accessAnalysis,
    llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap)
    : id(id), op(op), symbolTable(&symbolTable),
      accessAnalysis(&accessAnalysis), variablesMap(&variablesMap) {}
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
EquationTraits<EquationBridge *>::Id
EquationTraits<EquationBridge *>::getId(const Equation *equation) {
  return (*equation)->id;
}

size_t EquationTraits<EquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  auto numOfInductions =
      static_cast<uint64_t>((*equation)->op.getInductionVariables().size());

  if (numOfInductions == 0) {
    // Scalar equation.
    return 1;
  }

  return static_cast<size_t>(numOfInductions);
}

IndexSet
EquationTraits<EquationBridge *>::getIterationRanges(const Equation *equation) {
  IndexSet iterationSpace = (*equation)->op.getProperties().indices;

  if (iterationSpace.empty()) {
    // Scalar equation.
    iterationSpace += MultidimensionalRange(Range(0, 1));
  }

  return iterationSpace;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getAccesses(const Equation *equation) {
  std::vector<Access<VariableType, AccessProperty>> accesses;

  auto cachedAccesses =
      (*equation)->accessAnalysis->getAccesses(*(*equation)->symbolTable);

  if (cachedAccesses) {
    for (auto &access : *cachedAccesses) {
      auto accessFunction =
          getAccessFunction((*equation)->op.getContext(), access);

      auto variableIt = (*(*equation)->variablesMap).find(access.getVariable());

      if (variableIt != (*(*equation)->variablesMap).end()) {
        accesses.emplace_back(variableIt->getSecond(),
                              std::move(accessFunction), access.getPath());
      }
    }
  }

  return accesses;
}

llvm::raw_ostream &
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->op.printInline(os);
  return os;
}
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
EquationTraits<EquationBridge *>::Id
EquationTraits<EquationBridge *>::getId(const Equation *equation) {
  return (*equation)->id;
}

size_t EquationTraits<EquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  uint64_t numOfInductions =
      static_cast<uint64_t>((*equation)->op.getInductionVariables().size());

  if (numOfInductions == 0) {
    // Scalar equation.
    return 1;
  }

  return static_cast<size_t>(numOfInductions);
}

IndexSet
EquationTraits<EquationBridge *>::getIterationRanges(const Equation *equation) {
  IndexSet iterationSpace = (*equation)->op.getIterationSpace();

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

  if (mlir::failed(
          (*equation)->op.getAccesses(accesses, *(*equation)->symbolTable))) {
    return result;
  }

  for (VariableAccess &access : accesses) {
    auto accessFunction =
        getAccessFunction((*equation)->op.getContext(), access);

    auto variableIt = (*(*equation)->variablesMap).find(access.getVariable());

    if (variableIt != (*(*equation)->variablesMap).end()) {
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

  if (mlir::failed(
          (*equation)->op.getAccesses(accesses, *(*equation)->symbolTable))) {
    llvm_unreachable("Can't compute the accesses");
    return {};
  }

  llvm::SmallVector<VariableAccess> writeAccesses;

  if (mlir::failed((*equation)->op.getWriteAccesses(
          writeAccesses, *(*equation)->symbolTable,
          (*equation)->op.getProperties().indices, accesses))) {
    llvm_unreachable("Can't compute write accesses");
    return {};
  }

  std::vector<Access<VariableType, AccessProperty>> writes;

  for (const VariableAccess &access : writeAccesses) {
    auto variableIt = (*(*equation)->variablesMap).find(access.getVariable());

    writes.emplace_back(variableIt->getSecond(),
                        getAccessFunction((*equation)->op.getContext(), access),
                        access);
  }

  return writes;
}

std::vector<Access<EquationTraits<EquationBridge *>::VariableType,
                   EquationTraits<EquationBridge *>::AccessProperty>>
EquationTraits<EquationBridge *>::getReads(const Equation *equation) {
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(
          (*equation)->op.getAccesses(accesses, *(*equation)->symbolTable))) {
    llvm_unreachable("Can't compute the accesses");
    return {};
  }

  llvm::SmallVector<VariableAccess> readAccesses;

  if (mlir::failed((*equation)->op.getReadAccesses(
          readAccesses, *(*equation)->symbolTable,
          (*equation)->op.getProperties().indices, accesses))) {
    llvm_unreachable("Can't compute read accesses");
    return {};
  }

  std::vector<Access<VariableType, AccessProperty>> reads;

  for (const VariableAccess &access : readAccesses) {
    auto variableIt = (*(*equation)->variablesMap).find(access.getVariable());

    reads.emplace_back(variableIt->getSecond(),
                       getAccessFunction((*equation)->op.getContext(), access),
                       access);
  }

  return reads;
}

llvm::raw_ostream &
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->op.printInline(os);
  return os;
}
} // namespace marco::modeling::dependency
