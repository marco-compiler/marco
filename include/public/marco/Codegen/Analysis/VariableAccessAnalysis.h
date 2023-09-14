#ifndef MARCO_CODEGEN_ANALYSIS_VARIABLEACCESSANALYSIS_H
#define MARCO_CODEGEN_ANALYSIS_VARIABLEACCESSANALYSIS_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir::modelica
{
  class VariableAccessAnalysis
  {
    public:
      class AnalysisProvider
      {
        public:
          virtual std::optional<std::reference_wrapper<VariableAccessAnalysis>>
          getCachedVariableAccessAnalysis(EquationTemplateOp op) = 0;
      };

      class IRListener : public mlir::RewriterBase::Listener
      {
        public:
          IRListener(AnalysisProvider& provider);

          void notifyOperationRemoved(mlir::Operation* op) override;

        private:
          AnalysisProvider* provider;
      };

      VariableAccessAnalysis(EquationTemplateOp op);

      mlir::LogicalResult initialize(
          mlir::SymbolTableCollection& symbolTableCollection);

      /// Mark the analysis as to be preserved at the end of the pass.
      void preserve();

      /// Invalidate the analysis.
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
      bool initialized{false};
      bool valid{false};
      bool preserved{false};

      /// The list of all the accesses performed by the equation.
      llvm::DenseMap<
          uint64_t,
          llvm::SmallVector<VariableAccess, 10>> accesses;
  };
}

#endif // MARCO_CODEGEN_ANALYSIS_VARIABLEACCESSANALYSIS_H
