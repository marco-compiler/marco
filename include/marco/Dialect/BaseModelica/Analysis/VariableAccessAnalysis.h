#ifndef MARCO_DIALECT_BASEMODELICA_ANALYSIS_VARIABLEACCESSANALYSIS_H
#define MARCO_DIALECT_BASEMODELICA_ANALYSIS_VARIABLEACCESSANALYSIS_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir::bmodelica {
class VariableAccessAnalysis {
public:
  class AnalysisProvider {
  public:
    virtual ~AnalysisProvider();

    virtual std::optional<std::reference_wrapper<VariableAccessAnalysis>>
    getCachedVariableAccessAnalysis(EquationTemplateOp op) = 0;
  };

  class IRListener : public mlir::RewriterBase::Listener {
  public:
    explicit IRListener(AnalysisProvider &provider);

    void notifyOperationErased(mlir::Operation *op) override;

  private:
    AnalysisProvider *provider;
  };

  explicit VariableAccessAnalysis(EquationTemplateOp op);

  mlir::LogicalResult
  initialize(mlir::SymbolTableCollection &symbolTableCollection);

  /// Invalidate the analysis.
  void invalidate();

  /// Get the accesses.
  std::optional<llvm::ArrayRef<VariableAccess>>
  getAccesses(mlir::SymbolTableCollection &symbolTable);

private:
  mlir::LogicalResult
  loadAccesses(mlir::SymbolTableCollection &symbolTableCollection);

private:
  EquationTemplateOp equationTemplate;
  bool initialized{false};
  bool valid{false};

  /// The list of all the accesses performed by the equation.
  llvm::SmallVector<VariableAccess, 10> accesses;
};
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_ANALYSIS_VARIABLEACCESSANALYSIS_H
