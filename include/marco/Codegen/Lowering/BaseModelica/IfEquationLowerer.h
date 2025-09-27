#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_IFEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_IFEQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class IfEquationLowerer : public Lowerer {
public:
  explicit IfEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::IfEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_IFEQUATIONLOWERER_H
