#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_FORSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_FORSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class ForStatementLowerer : public Lowerer {
public:
  explicit ForStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::ForStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_FORSTATEMENTLOWERER_H
