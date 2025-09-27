#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_MODELLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_MODELLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class ModelLowerer : public Lowerer {
public:
  explicit ModelLowerer(BridgeInterface *bridge);

  void declare(const ast::bmodelica::Model &model) override;

  [[nodiscard]] bool
  declareVariables(const ast::bmodelica::Model &model) override;

  [[nodiscard]] bool lower(const ast::bmodelica::Model &model) override;

  [[nodiscard]] bool
  lowerVariableAttributes(mlir::bmodelica::ModelOp modelOp,
                          const ast::bmodelica::Member &variable);

  [[nodiscard]] bool lowerVariableAttributes(
      mlir::bmodelica::ModelOp modelOp,
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &components,
      const ast::bmodelica::ClassModification &classModification);

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_MODELLOWERER_H
