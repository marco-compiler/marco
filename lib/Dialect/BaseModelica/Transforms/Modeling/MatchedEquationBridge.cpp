#include "marco/Dialect/BaseModelica/Transforms/Modeling/MatchedEquationBridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
MatchedEquationBridge::MatchedEquationBridge(
    int64_t id, EquationInstanceOp op, mlir::SymbolTableCollection &symbolTable,
    VariableAccessAnalysis &accessAnalysis,
    llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap)
    : id(id), op(op), symbolTable(&symbolTable),
      accessAnalysis(&accessAnalysis), variablesMap(&variablesMap) {}
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::dependency {
EquationTraits<MatchedEquationBridge *>::Id
EquationTraits<MatchedEquationBridge *>::getId(const Equation *equation) {
  return (*equation)->id;
}

size_t EquationTraits<MatchedEquationBridge *>::getNumOfIterationVars(
    const Equation *equation) {
  uint64_t numOfInductions =
      static_cast<uint64_t>((*equation)->op.getInductionVariables().size());

  if (numOfInductions == 0) {
    // Scalar equation.
    return 1;
  }

  return static_cast<size_t>(numOfInductions);
}

IndexSet EquationTraits<MatchedEquationBridge *>::getIterationRanges(
    const Equation *equation) {
  IndexSet iterationSpace = (*equation)->op.getIterationSpace();

  if (iterationSpace.empty()) {
    // Scalar equation.
    iterationSpace += MultidimensionalRange(Range(0, 1));
  }

  return iterationSpace;
}

std::vector<Access<EquationTraits<MatchedEquationBridge *>::VariableType,
                   EquationTraits<MatchedEquationBridge *>::AccessProperty>>
EquationTraits<MatchedEquationBridge *>::getAccesses(const Equation *equation) {
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

Access<EquationTraits<MatchedEquationBridge *>::VariableType,
       EquationTraits<MatchedEquationBridge *>::AccessProperty>
EquationTraits<MatchedEquationBridge *>::getWrite(const Equation *equation) {
  EquationInstanceOp equationOp = (*equation)->op;
  auto &symbolTableCollection = *(*equation)->symbolTable;

  llvm::SmallVector<VariableAccess> accesses;
  equationOp.getAccesses(accesses, symbolTableCollection);

  llvm::SmallVector<VariableAccess> writeAccesses;
  equationOp.getWriteAccesses(writeAccesses, symbolTableCollection, accesses);

  assert(!writeAccesses.empty());

  auto accessFunction =
      getAccessFunction((*equation)->op.getContext(), writeAccesses[0]);

  return {(*(*equation)->variablesMap)[writeAccesses[0].getVariable()],
          std::move(accessFunction), writeAccesses[0]};
}

std::vector<Access<EquationTraits<MatchedEquationBridge *>::VariableType,
                   EquationTraits<MatchedEquationBridge *>::AccessProperty>>
EquationTraits<MatchedEquationBridge *>::getReads(const Equation *equation) {
  IndexSet equationIndices = getIterationRanges(equation);

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(
          (*equation)->op.getAccesses(accesses, *(*equation)->symbolTable))) {
    llvm_unreachable("Can't compute the accesses");
    return {};
  }

  llvm::SmallVector<VariableAccess> readAccesses;

  if (mlir::failed((*equation)->op.getReadAccesses(
          readAccesses, *(*equation)->symbolTable, equationIndices,
          accesses))) {
    llvm_unreachable("Can't compute read accesses");
    return {};
  }

  std::vector<Access<VariableType, AccessProperty>> reads;

  for (const VariableAccess &readAccess : readAccesses) {
    auto variableIt =
        (*(*equation)->variablesMap).find(readAccess.getVariable());

    reads.emplace_back(
        variableIt->getSecond(),
        getAccessFunction((*equation)->op.getContext(), readAccess),
        readAccess);
  }

  return reads;
}

std::unique_ptr<AccessFunction>
EquationTraits<MatchedEquationBridge *>::getAccessFunction(
    mlir::MLIRContext *context, const VariableAccess &access) {
  const AccessFunction &accessFunction = access.getAccessFunction();

  if (accessFunction.getNumOfResults() == 0) {
    // Access to scalar variable.
    return AccessFunction::build(
        mlir::AffineMap::get(accessFunction.getNumOfDims(), 0,
                             mlir::getAffineConstantExpr(0, context)));
  }

  return accessFunction.clone();
}

llvm::raw_ostream &
EquationTraits<MatchedEquationBridge *>::dump(const Equation *equation,
                                              llvm::raw_ostream &os) {
  (*equation)->op.printInline(os);
  return os;
}
} // namespace marco::modeling::dependency
