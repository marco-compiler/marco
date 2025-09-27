#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUATIONSECTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUATIONSECTIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class EquationSectionLowerer : public Lowerer {
public:
  explicit EquationSectionLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::EquationSection &equationSection) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUATIONSECTIONLOWERER_H
