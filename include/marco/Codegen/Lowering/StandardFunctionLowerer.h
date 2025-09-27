#ifndef MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class StandardFunctionLowerer : public Lowerer {
public:
  explicit StandardFunctionLowerer(BridgeInterface *bridge);

  void declare(const ast::bmodelica::StandardFunction &function) override;

  [[nodiscard]] bool
  declareVariables(const ast::bmodelica::StandardFunction &function) override;

  [[nodiscard]] bool
  lower(const ast::bmodelica::StandardFunction &function) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;

private:
  [[nodiscard]] bool
  lowerVariableDefaultValue(const ast::bmodelica::Member &variable);

  bool isRecordConstructor(const ast::bmodelica::StandardFunction &function);

  [[nodiscard]] bool lowerExternalFunctionCall(
      llvm::StringRef language,
      const ast::bmodelica::ExternalFunctionCall &externalFunctionCall,
      mlir::bmodelica::FunctionOp functionOp);

  [[nodiscard]] bool
  createImplicitExternalFunctionCall(const ast::bmodelica::Function &function);
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONLOWERER_H
