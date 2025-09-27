#ifndef MARCO_CODEGEN_LOWERING_SUBSCRIPTLOWERER_H
#define MARCO_CODEGEN_LOWERING_SUBSCRIPTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class SubscriptLowerer : public Lowerer {
public:
  explicit SubscriptLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::Subscript &subscript) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_SUBSCRIPTLOWERER_H
