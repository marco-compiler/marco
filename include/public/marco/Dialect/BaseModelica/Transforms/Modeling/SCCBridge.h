#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_SCCBRIDGE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_SCCBRIDGE_H

#include "marco/Dialect/BaseModelica/IR/BaseModelicaDialect.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/MatchedEquationBridge.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::bmodelica::bridge
{
  class SCCBridge
  {
    public:
      SCCOp op;
      mlir::SymbolTableCollection* symbolTable;
      WritesMap<VariableOp, MatchedEquationInstanceOp>* matchedEqsWritesMap;
      WritesMap<VariableOp, StartEquationInstanceOp>* startEqsWritesMap;

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
          WritesMap<VariableOp, MatchedEquationInstanceOp>& matchedEqsWritesMap,
          WritesMap<VariableOp, StartEquationInstanceOp>& startEqsWritesMap,
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
  struct SCCTraits<::mlir::bmodelica::bridge::SCCBridge*>
  {
    using SCC = ::mlir::bmodelica::bridge::SCCBridge*;
    using ElementRef = ::mlir::bmodelica::bridge::MatchedEquationBridge*;

    static std::vector<ElementRef> getElements(const SCC* scc);

    static std::vector<ElementRef>
    getDependencies(const SCC* scc, ElementRef equation);
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_SCCBRIDGE_H
