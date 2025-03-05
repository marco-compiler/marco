#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

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

std::unique_ptr<AccessFunction>
EquationTraits<EquationBridge *>::getAccessFunction(
    mlir::MLIRContext *context, const mlir::bmodelica::VariableAccess &access) {
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
EquationTraits<EquationBridge *>::dump(const Equation *equation,
                                       llvm::raw_ostream &os) {
  (*equation)->op.printInline(os);
  return os;
}
} // namespace marco::modeling::matching
