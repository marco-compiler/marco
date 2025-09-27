#ifndef MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class PartialDerFunctionLowerer : public Lowerer {
public:
  explicit PartialDerFunctionLowerer(BridgeInterface *bridge);

  void declare(const ast::bmodelica::PartialDerFunction &function) override;

  [[nodiscard]] bool
  declareVariables(const ast::bmodelica::PartialDerFunction &function) override;

  [[nodiscard]] bool
  lower(const ast::bmodelica::PartialDerFunction &function) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H
