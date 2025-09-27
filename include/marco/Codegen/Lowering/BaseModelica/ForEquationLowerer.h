#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_FOREQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_FOREQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class ForEquationLowerer : public Lowerer {
public:
  explicit ForEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::ForEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_FOREQUATIONLOWERER_H
