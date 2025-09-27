#ifndef MARCO_CODEGEN_LOWERING_BREAKSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BREAKSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class BreakStatementLowerer : public Lowerer {
public:
  explicit BreakStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::BreakStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_BREAKSTATEMENTLOWERER_H
