#ifndef MARCO_CODEGEN_LOWERING_IFEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_IFEQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class IfEquationLowerer : public Lowerer {
public:
  explicit IfEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::IfEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_IFEQUATIONLOWERER_H
