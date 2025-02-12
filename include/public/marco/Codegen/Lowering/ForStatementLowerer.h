#ifndef MARCO_CODEGEN_LOWERING_FORSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_FORSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class ForStatementLowerer : public Lowerer {
public:
  explicit ForStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::ForStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_FORSTATEMENTLOWERER_H
