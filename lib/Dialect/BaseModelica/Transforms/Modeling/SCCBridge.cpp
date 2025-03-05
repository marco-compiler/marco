#include "marco/Dialect/BaseModelica/Transforms/Modeling/SCCBridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
SCCBridge::SCCBridge(
    SCCOp op, mlir::SymbolTableCollection &symbolTable,
    WritesMap<VariableOp, EquationInstanceOp> &matchedEqsWritesMap,
    WritesMap<VariableOp, StartEquationInstanceOp> &startEqsWritesMap,
    llvm::DenseMap<EquationInstanceOp, MatchedEquationBridge *> &equationsMap)
    : op(op), symbolTable(&symbolTable),
      matchedEqsWritesMap(&matchedEqsWritesMap),
      startEqsWritesMap(&startEqsWritesMap), equationsMap(&equationsMap) {}
} // namespace mlir::bmodelica::bridge

namespace {
template <typename Equation>
static mlir::LogicalResult
getWritingEquations(llvm::SmallVectorImpl<Equation> &result,
                    const WritesMap<VariableOp, Equation> &writesMap,
                    VariableOp variable, const IndexSet &variableIndices) {
  auto writingEquations = writesMap.equal_range(variable);

  for (const auto &writingEquation :
       llvm::make_range(writingEquations.first, writingEquations.second)) {
    if (variableIndices.empty()) {
      result.push_back(writingEquation.second.second);
    } else {
      const IndexSet &writtenVariableIndices = writingEquation.second.first;

      if (writtenVariableIndices.overlaps(variableIndices)) {
        result.push_back(writingEquation.second.second);
      }
    }
  }

  return mlir::success();
}
} // namespace

namespace marco::modeling::dependency {
std::vector<SCCTraits<SCCBridge *>::ElementRef>
SCCTraits<SCCBridge *>::getElements(const SCC *scc) {
  mlir::bmodelica::SCCOp sccOp = (*scc)->op;
  const auto &equationsMap = (*scc)->equationsMap;
  std::vector<ElementRef> result;

  for (mlir::bmodelica::EquationInstanceOp equation :
       sccOp.getOps<mlir::bmodelica::EquationInstanceOp>()) {
    ElementRef equationPtr = equationsMap->lookup(equation);
    assert(equationPtr && "Equation bridge not found");
    result.push_back(equationPtr);
  }

  return result;
}

std::vector<SCCTraits<SCCBridge *>::ElementRef>
SCCTraits<SCCBridge *>::getDependencies(const SCC *scc, ElementRef equation) {
  mlir::SymbolTableCollection &symbolTableCollection = *equation->symbolTable;

  const auto &accesses =
      equation->accessAnalysis->getAccesses(symbolTableCollection);

  if (!accesses) {
    llvm_unreachable("Can't obtain accesses");
    return {};
  }

  llvm::SmallVector<mlir::bmodelica::VariableAccess> readAccesses;

  if (mlir::failed(equation->op.getReadAccesses(
          readAccesses, symbolTableCollection, *accesses))) {
    llvm_unreachable("Can't obtain read accesses");
    return {};
  }

  IndexSet equationIndices = equation->op.getIterationSpace();
  auto modelOp = (*scc)->op->getParentOfType<mlir::bmodelica::ModelOp>();
  const auto &matchedEqsWritesMap = *(*scc)->matchedEqsWritesMap;
  const auto &startEqsWritesMap = *(*scc)->startEqsWritesMap;
  const auto &equationsMap = *(*scc)->equationsMap;

  std::vector<ElementRef> result;

  for (const mlir::bmodelica::VariableAccess &readAccess : readAccesses) {
    auto variableOp =
        symbolTableCollection.lookupSymbolIn<mlir::bmodelica::VariableOp>(
            modelOp, readAccess.getVariable());

    assert(variableOp && "Variable not found");

    IndexSet readVariableIndices =
        readAccess.getAccessFunction().map(equationIndices);

    llvm::SmallVector<EquationInstanceOp> writingEquations;

    if (mlir::failed(getWritingEquations(writingEquations, matchedEqsWritesMap,
                                         variableOp, readVariableIndices))) {
      llvm_unreachable("Can't determine the writing equations");
      return {};
    }

    for (const auto &writingEquation : writingEquations) {
      if (auto writingEquationPtr = equationsMap.lookup(writingEquation)) {
        result.push_back(writingEquationPtr);
      }
    }
  }

  llvm::SmallVector<VariableAccess> writeAccesses;

  if (mlir::failed(equation->op.getWriteAccesses(
          writeAccesses, symbolTableCollection, *accesses))) {
    llvm_unreachable("Can't determine the write accesses");
    return {};
  }

  llvm::SmallVector<StartEquationInstanceOp> startEquations;

  for (const auto &writeAccess : writeAccesses) {
    auto writtenVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, writeAccess.getVariable());

    assert(writtenVariableOp != nullptr);

    IndexSet writtenVariableIndices =
        writeAccess.getAccessFunction().map(equationIndices);

    if (mlir::failed(getWritingEquations(startEquations, startEqsWritesMap,
                                         writtenVariableOp,
                                         writtenVariableIndices))) {
      llvm_unreachable("Can't determine the writing equations");
      return {};
    }
  }

  for (StartEquationInstanceOp startEquation : startEquations) {
    IndexSet startEquationIndices = startEquation.getIterationSpace();

    llvm::SmallVector<mlir::bmodelica::VariableAccess, 10> startEqAccesses;

    if (mlir::failed(startEquation.getAccesses(startEqAccesses,
                                               symbolTableCollection))) {
      llvm_unreachable("Can't compute accesses");
      return {};
    }

    llvm::SmallVector<mlir::bmodelica::VariableAccess, 10> startEqReads;

    if (mlir::failed(startEquation.getReadAccesses(
            startEqReads, symbolTableCollection, startEqAccesses))) {
      llvm_unreachable("Can't compute read accesses");
      return {};
    }

    for (const auto &startEqRead : startEqReads) {
      auto readVariableOp =
          symbolTableCollection.lookupSymbolIn<mlir::bmodelica::VariableOp>(
              modelOp, startEqRead.getVariable());

      assert(readVariableOp != nullptr);

      IndexSet readVariableIndices =
          startEqRead.getAccessFunction().map(startEquationIndices);

      llvm::SmallVector<EquationInstanceOp> writingEquations;

      if (mlir::failed(getWritingEquations(writingEquations,
                                           matchedEqsWritesMap, readVariableOp,
                                           readVariableIndices))) {
        llvm_unreachable("Can't determine the writing equations");
        return {};
      }

      for (const auto &writingEquation : writingEquations) {
        if (auto writingEquationPtr = equationsMap.lookup(writingEquation)) {
          result.push_back(writingEquationPtr);
        }
      }
    }
  }

  return result;
}

llvm::raw_ostream &SCCTraits<SCCBridge *>::dump(const SCC *scc,
                                                llvm::raw_ostream &os) {
  return os << "not implemented";
}
} // namespace marco::modeling::dependency
