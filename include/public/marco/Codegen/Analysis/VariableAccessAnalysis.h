#ifndef MARCO_CODEGEN_ANALYSIS_VARIABLEACCESSANALYSIS_H
#define MARCO_CODEGEN_ANALYSIS_VARIABLEACCESSANALYSIS_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir::modelica
{
  class VariableAccessAnalysis
  {
    public:
      VariableAccessAnalysis(EquationTemplateOp op);

      mlir::LogicalResult initialize(
          mlir::SymbolTableCollection& symbolTableCollection);

      void invalidate();

      bool isInvalidated(const mlir::detail::PreservedAnalyses& pa) const;

      /// Get the accesses of an equation.
      /// Returns std::nullopt if the accesses can't be computed.
      std::optional<llvm::ArrayRef<VariableAccess>> getAccesses(
          EquationInstanceOp instanceOp,
          mlir::SymbolTableCollection& symbolTable);

      /// Get the accesses of a matched equation.
      /// Returns std::nullopt if the accesses can't be computed.
      std::optional<llvm::ArrayRef<VariableAccess>> getAccesses(
          MatchedEquationInstanceOp instanceOp,
          mlir::SymbolTableCollection& symbolTable);

      /// Get the accesses of a scheduled equation.
      /// Returns std::nullopt if the accesses can't be computed.
      std::optional<llvm::ArrayRef<VariableAccess>> getAccesses(
          ScheduledEquationInstanceOp instanceOp,
          mlir::SymbolTableCollection& symbolTable);

    private:
      mlir::LogicalResult loadAccesses(
        mlir::SymbolTableCollection& symbolTableCollection);

    private:
      EquationTemplateOp equationTemplate;
      bool initialized;
      bool valid;

      /// The list of all the accesses performed by the equation.
      llvm::DenseMap<
          uint64_t,
          llvm::SmallVector<VariableAccess, 10>> accesses;
  };
}

#endif // MARCO_CODEGEN_ANALYSIS_VARIABLEACCESSANALYSIS_H
