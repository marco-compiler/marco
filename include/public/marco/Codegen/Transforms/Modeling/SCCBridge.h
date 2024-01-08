#ifndef MARCO_CODEGEN_TRANSFORMS_MODELING_SCCBRIDGE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELING_SCCBRIDGE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/VariableBridge.h"
#include "marco/Codegen/Transforms/Modeling/MatchedEquationBridge.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::modelica::bridge
{
  class SCCBridge
  {
    public:
      SCCOp op;
      mlir::SymbolTableCollection* symbolTable;
      WritesMap<SimulationVariableOp, MatchedEquationInstanceOp>* writesMap;

      llvm::DenseMap<
          MatchedEquationInstanceOp, MatchedEquationBridge*>* equationsMap;

    public:
      template<typename... Args>
      static std::unique_ptr<SCCBridge> build(Args&&... args)
      {
        return std::make_unique<SCCBridge>(
            std::forward<Args>(args)...);
      }

      SCCBridge(
          SCCOp op,
          mlir::SymbolTableCollection& symbolTable,
          WritesMap<
              SimulationVariableOp, MatchedEquationInstanceOp>& writesMap,
          llvm::DenseMap<
              MatchedEquationInstanceOp, MatchedEquationBridge*>& equationsMap);

      // Forbid copies to avoid dangling pointers by design.
      SCCBridge(const SCCBridge& other) = delete;
      SCCBridge(SCCBridge&& other) = delete;
      SCCBridge& operator=(const SCCBridge& other) = delete;
      SCCBridge& operator==(const SCCBridge& other) = delete;
  };
}

namespace marco::modeling::dependency
{
  template<>
  struct SCCTraits<::mlir::modelica::bridge::SCCBridge*>
  {
    using SCC = ::mlir::modelica::bridge::SCCBridge*;
    using ElementRef = ::mlir::modelica::bridge::MatchedEquationBridge*;

    static std::vector<ElementRef> getElements(const SCC* scc)
    {
      mlir::modelica::SCCOp sccOp = (*scc)->op;
      const auto& equationsMap = (*scc)->equationsMap;
      std::vector<ElementRef> result;

      for (mlir::modelica::MatchedEquationInstanceOp equation :
           sccOp.getOps<mlir::modelica::MatchedEquationInstanceOp>()) {
        ElementRef equationPtr = equationsMap->lookup(equation);
        assert(equationPtr && "Equation bridge not found");
        result.push_back(equationPtr);
      }

      return result;
    }

    static std::vector<ElementRef> getDependencies(
        const SCC* scc, ElementRef equation)
    {
      mlir::SymbolTableCollection& symbolTableCollection =
          *equation->symbolTable;

      const auto& accesses = equation->accessAnalysis->getAccesses(
          equation->op, symbolTableCollection);

      if (!accesses) {
        llvm_unreachable("Can't obtain accesses");
        return {};
      }

      auto matchedAccess = equation->op.getMatchedAccess(symbolTableCollection);

      if (!matchedAccess) {
        llvm_unreachable("Can't obtain matched access");
        return {};
      }

      llvm::SmallVector<mlir::modelica::VariableAccess> readAccesses;

      if (mlir::failed(equation->op.getReadAccesses(
              readAccesses, symbolTableCollection, *accesses))) {
        llvm_unreachable("Can't obtain read accesses");
        return {};
      }

      IndexSet equationIndices = equation->op.getIterationSpace();
      auto moduleOp = (*scc)->op->getParentOfType<mlir::ModuleOp>();
      const auto& writesMap = (*scc)->writesMap;
      const auto& equationMap = (*scc)->equationsMap;

      std::vector<ElementRef> result;

      for (const mlir::modelica::VariableAccess& readAccess : readAccesses) {
        auto simulationVariableOp =
            symbolTableCollection.lookupSymbolIn<
                mlir::modelica::SimulationVariableOp>(
                moduleOp, readAccess.getVariable());

        IndexSet readVariableIndices =
            readAccess.getAccessFunction().map(equationIndices);

        auto writingEquations = writesMap->equal_range(simulationVariableOp);

        for (const auto& writingEquation : llvm::make_range(
                 writingEquations.first, writingEquations.second)) {
          if (auto writingEquationPtr =
                  equationMap->lookup(writingEquation.second.second)) {
            if (readVariableIndices.empty()) {
              result.push_back(writingEquationPtr);
            } else {
              const IndexSet& writtenVariableIndices = writingEquation.second.first;

              if (writtenVariableIndices.overlaps(readVariableIndices)) {
                result.push_back(writingEquationPtr);
              }
            }
          }
        }
      }

      return result;
    }
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELING_SCCBRIDGE_H
