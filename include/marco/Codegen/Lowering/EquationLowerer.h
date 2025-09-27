#ifndef MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class EquationLowerer : public Lowerer {
public:
  explicit EquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::bmodelica::Equation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H
