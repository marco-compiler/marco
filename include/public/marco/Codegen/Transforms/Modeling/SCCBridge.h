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
  struct SCCTraits<::mlir::modelica::bridge::SCCBridge*>
  {
    using SCC = ::mlir::modelica::bridge::SCCBridge*;
    using ElementRef = ::mlir::modelica::bridge::MatchedEquationBridge*;

    static std::vector<ElementRef> getElements(const SCC* scc);

    static std::vector<ElementRef> getDependencies(
        const SCC* scc, ElementRef equation);
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELING_SCCBRIDGE_H
