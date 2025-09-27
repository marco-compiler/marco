#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_SUBSCRIPTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_SUBSCRIPTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class SubscriptLowerer : public Lowerer {
public:
  explicit SubscriptLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::Subscript &subscript) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_SUBSCRIPTLOWERER_H
