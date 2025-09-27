#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_IFSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_IFSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class IfStatementLowerer : public Lowerer {
public:
  explicit IfStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::IfStatement &statement) override;

protected:
  using Lowerer::lower;

private:
  std::optional<mlir::Value>
  lowerCondition(const ast::bmodelica::Expression &expression);

  [[nodiscard]] bool
  lower(const ast::bmodelica::StatementsBlock &statementsBlock);
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_IFSTATEMENTLOWERER_H
