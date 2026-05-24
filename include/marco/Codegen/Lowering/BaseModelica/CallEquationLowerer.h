#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLEQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class CallEquationLowerer : public Lowerer {
public:
  explicit CallEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] virtual bool
  lower(const ast::bmodelica::CallEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLEQUATIONLOWERER_H
