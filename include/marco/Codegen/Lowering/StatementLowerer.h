#ifndef MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class StatementLowerer : public Lowerer {
public:
  explicit StatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::bmodelica::Statement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H
