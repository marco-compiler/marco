#ifndef MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class PartialDerFunctionLowerer : public Lowerer {
public:
  explicit PartialDerFunctionLowerer(BridgeInterface *bridge);

  void declare(const ast::PartialDerFunction &function) override;

  [[nodiscard]] bool
  declareVariables(const ast::PartialDerFunction &function) override;

  [[nodiscard]] bool lower(const ast::PartialDerFunction &function) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H
