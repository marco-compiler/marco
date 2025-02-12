#ifndef MARCO_CODEGEN_LOWERING_EXPRESSIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EXPRESSIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/Results.h"

namespace marco::codegen::lowering {
class ExpressionLowerer : public Lowerer {
public:
  explicit ExpressionLowerer(BridgeInterface *bridge);

  std::optional<Results> lower(const ast::Expression &expression) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_EXPRESSIONLOWERER_H
