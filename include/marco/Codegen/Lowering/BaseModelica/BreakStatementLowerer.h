#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_BREAKSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_BREAKSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class BreakStatementLowerer : public Lowerer {
public:
  explicit BreakStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::BreakStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_BREAKSTATEMENTLOWERER_H
