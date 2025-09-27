#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUALITYEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUALITYEQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class EqualityEquationLowerer : public Lowerer {
public:
  explicit EqualityEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::EqualityEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_EQUALITYEQUATIONLOWERER_H
