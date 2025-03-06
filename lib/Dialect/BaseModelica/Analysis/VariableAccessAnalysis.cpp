#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
VariableAccessAnalysis::AnalysisProvider::~AnalysisProvider() = default;

VariableAccessAnalysis::VariableAccessAnalysis(EquationTemplateOp op)
    : equationTemplate(op) {}

mlir::LogicalResult VariableAccessAnalysis::initialize(
    mlir::SymbolTableCollection &symbolTableCollection) {
  if (initialized) {
    return mlir::success();
  }

  if (mlir::failed(loadAccesses(symbolTableCollection))) {
    initialized = false;
    return mlir::failure();
  }

  initialized = true;
  return mlir::success();
}

void VariableAccessAnalysis::invalidate() {
  valid = false;
  accesses.clear();
}

std::optional<llvm::ArrayRef<VariableAccess>>
VariableAccessAnalysis::getAccesses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  assert(initialized && "Variable access analysis not initialized");

  if (!valid) {
    if (mlir::failed(loadAccesses(symbolTableCollection))) {
      return std::nullopt;
    }
  }

  return accesses;
}

mlir::LogicalResult VariableAccessAnalysis::loadAccesses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  if (mlir::failed(
          equationTemplate.getAccesses(accesses, symbolTableCollection))) {
    return mlir::failure();
  }

  valid = true;
  return mlir::success();
}

VariableAccessAnalysis::IRListener::IRListener(AnalysisProvider &provider)
    : provider(&provider) {}

void VariableAccessAnalysis::IRListener::notifyOperationErased(Operation *op) {
  Listener::notifyOperationErased(op);

  if (auto equationOp = mlir::dyn_cast<EquationInstanceOp>(op)) {
    auto analysis =
        provider->getCachedVariableAccessAnalysis(equationOp.getTemplate());

    if (analysis) {
      analysis->get().invalidate();
    }
  }
}
} // namespace mlir::bmodelica
