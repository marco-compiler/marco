#ifndef MARCO_CODEGEN_LOWERING_CALLLOWERER_H
#define MARCO_CODEGEN_LOWERING_CALLLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"
#include <functional>

namespace marco::codegen::lowering {
class ExpressionLowerer;

class ExternalFunctionCallLowerer : public Lowerer {
public:
  explicit ExternalFunctionCallLowerer(BridgeInterface *bridge);

  virtual std::optional<Results> lower(const ast::ExternalFunctionCall &call) override;

protected:
  using Lowerer::lower;

private:
  std::optional<mlir::Operation *>
  resolveCallee(llvm::StringRef calleeName);

  std::optional<mlir::Value> lowerArg(const ast::Expression &expression);

  void getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
      mlir::bmodelica::FunctionOp functionOp);


  [[nodiscard]] bool lowerCustomFunctionArgs(
      const ast::ExternalFunctionCall &call,
      llvm::ArrayRef<mlir::bmodelica::VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string> &argNames,
      llvm::SmallVectorImpl<mlir::Value> &argValues);

};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_CALLLOWERER_H
