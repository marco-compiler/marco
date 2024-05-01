#ifndef MARCO_CODEGEN_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELING_EQUATIONBRIDGE_H

#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/VariableBridge.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::bmodelica::bridge
{
  class EquationBridge
  {
    public:
      EquationInstanceOp op;
      mlir::SymbolTableCollection* symbolTable;
      VariableAccessAnalysis* accessAnalysis;
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>* variablesMap;

    public:
      template<typename... Args>
      static std::unique_ptr<EquationBridge> build(Args&&... args)
      {
        return std::make_unique<EquationBridge>(std::forward<Args>(args)...);
      }

      EquationBridge(
          EquationInstanceOp op,
          mlir::SymbolTableCollection& symbolTable,
          VariableAccessAnalysis& accessAnalysis,
          llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap);

      // Forbid copies to avoid dangling pointers by design.
      EquationBridge(const EquationBridge& other) = delete;
      EquationBridge(EquationBridge&& other) = delete;
      EquationBridge& operator=(const EquationBridge& other) = delete;
      EquationBridge& operator==(const EquationBridge& other) = delete;
  };
}

namespace marco::modeling::matching
{
  template<>
  struct EquationTraits<::mlir::bmodelica::bridge::EquationBridge*>
  {
    using Equation = ::mlir::bmodelica::bridge::EquationBridge*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation);

    static size_t getNumOfIterationVars(const Equation* equation);

    static IndexSet getIterationRanges(const Equation* equation);

    using VariableType = ::mlir::bmodelica::bridge::VariableBridge*;
    using AccessProperty = ::mlir::bmodelica::EquationPath;

    static std::vector<Access<VariableType, AccessProperty>>
    getAccesses(const Equation* equation);

    static std::unique_ptr<AccessFunction> getAccessFunction(
        mlir::MLIRContext* context,
        const mlir::bmodelica::VariableAccess& access);
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
