#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_MATCHEDEQUATIONBRIDGE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_MATCHEDEQUATIONBRIDGE_H

#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::bmodelica::bridge
{
  class MatchedEquationBridge
  {
    public:
      MatchedEquationInstanceOp op;
      mlir::SymbolTableCollection* symbolTable;
      VariableAccessAnalysis* accessAnalysis;
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>* variablesMap;

    public:
      template<typename... Args>
      static std::unique_ptr<MatchedEquationBridge> build(Args&&... args)
      {
        return std::make_unique<MatchedEquationBridge>(
            std::forward<Args>(args)...);
      }

      MatchedEquationBridge(
          MatchedEquationInstanceOp op,
          mlir::SymbolTableCollection& symbolTable,
          VariableAccessAnalysis& accessAnalysis,
          llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap);

      // Forbid copies to avoid dangling pointers by design.
      MatchedEquationBridge(const MatchedEquationBridge& other) = delete;
      MatchedEquationBridge(MatchedEquationBridge&& other) = delete;

      MatchedEquationBridge& operator=(
          const MatchedEquationBridge& other) = delete;

      MatchedEquationBridge& operator==(
          const MatchedEquationBridge& other) = delete;
  };
}

namespace marco::modeling::dependency
{
  template<>
  struct EquationTraits<::mlir::bmodelica::bridge::MatchedEquationBridge*>
  {
    using Equation = ::mlir::bmodelica::bridge::MatchedEquationBridge*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation);

    static size_t getNumOfIterationVars(const Equation* equation);

    static IndexSet getIterationRanges(const Equation* equation);

    using VariableType = ::mlir::bmodelica::bridge::VariableBridge*;
    using VariableAccess = mlir::bmodelica::VariableAccess;
    using AccessProperty = VariableAccess;

    static std::vector<Access<VariableType, AccessProperty>>
    getAccesses(const Equation* equation);

    static Access<VariableType, AccessProperty>
    getWrite(const Equation* equation);

    static std::vector<Access<VariableType, AccessProperty>>
    getReads(const Equation* equation);

    static std::unique_ptr<AccessFunction> getAccessFunction(
        mlir::MLIRContext* context,
        const VariableAccess& access);
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_MATCHEDEQUATIONBRIDGE_H
