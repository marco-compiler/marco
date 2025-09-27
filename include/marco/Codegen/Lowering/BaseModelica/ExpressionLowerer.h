#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_EXPRESSIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_EXPRESSIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/Results.h"

namespace marco::codegen::lowering::bmodelica {
class ExpressionLowerer : public Lowerer {
public:
  explicit ExpressionLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::Expression &expression) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_EXPRESSIONLOWERER_H
