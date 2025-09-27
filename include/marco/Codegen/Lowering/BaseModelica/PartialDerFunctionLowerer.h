#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_PARTIALDERFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_PARTIALDERFUNCTIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
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
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_PARTIALDERFUNCTIONLOWERER_H
