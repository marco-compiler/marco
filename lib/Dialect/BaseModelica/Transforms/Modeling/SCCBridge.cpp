#include "marco/Dialect/BaseModelica/Transforms/Modeling/SCCBridge.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace mlir::bmodelica::bridge {
SCCBridge::SCCBridge(
    SCCOp op, mlir::SymbolTableCollection &symbolTable,
    WritesMap<VariableOp, EquationInstanceOp> &matchedEqsWritesMap,
    WritesMap<VariableOp, StartEquationInstanceOp> &startEqsWritesMap,
    llvm::DenseMap<EquationInstanceOp, EquationBridge *> &equationsMap)
    : op(op), symbolTable(&symbolTable),
      matchedEqsWritesMap(&matchedEqsWritesMap),
      startEqsWritesMap(&startEqsWritesMap), equationsMap(&equationsMap) {}
} // namespace mlir::bmodelica::bridge

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
  mlir::SymbolTableCollection &symbolTableCollection =
      equation->getSymbolTableCollection();

  llvm::ArrayRef<VariableAccess> accessesRef;
  bool cached = false;

  if (equation->hasAccessAnalysis()) {
    if (auto cachedAccesses =
            equation->getAccessAnalysis().getAccesses(symbolTableCollection)) {
      accessesRef = *cachedAccesses;
      cached = true;
    }
  }

  llvm::SmallVector<VariableAccess> accesses;

  if (!cached) {
    if (mlir::failed(
            equation->getOp().getAccesses(accesses, symbolTableCollection))) {
      llvm_unreachable("Can't compute the accesses");
      return {};
    }

    accessesRef = accesses;
  }

  llvm::SmallVector<mlir::bmodelica::VariableAccess> readAccesses;

  if (mlir::failed(equation->getOp().getReadAccesses(
          readAccesses, symbolTableCollection, accessesRef))) {
    llvm_unreachable("Can't obtain read accesses");
    return {};
  }

  IndexSet equationIndices = equation->getOp().getIterationSpace();
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

    auto writingEquations =
        matchedEqsWritesMap.getWrites(variableOp, readVariableIndices);

    for (const auto &writingEquation : writingEquations) {
      if (auto writingEquationPtr =
              equationsMap.lookup(writingEquation.writingEntity)) {
        result.push_back(writingEquationPtr);
      }
    }
  }

  llvm::SmallVector<VariableAccess> writeAccesses;

  if (mlir::failed(equation->getOp().getWriteAccesses(
          writeAccesses, symbolTableCollection, accessesRef))) {
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

    auto writingEquations =
        startEqsWritesMap.getWrites(writtenVariableOp, writtenVariableIndices);

    for (const auto &writingEquation : writingEquations) {
      startEquations.push_back(writingEquation.writingEntity);
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

      auto writingEquations =
          matchedEqsWritesMap.getWrites(readVariableOp, readVariableIndices);

      for (const auto &writingEquation : writingEquations) {
        if (auto writingEquationPtr =
                equationsMap.lookup(writingEquation.writingEntity)) {
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
