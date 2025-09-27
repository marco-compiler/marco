#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class EquationLowerer : public Lowerer {
public:
  explicit EquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::bmodelica::Equation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUATIONLOWERER_H
