#ifndef MARCO_CODEGEN_LOWERING_RETURNSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_RETURNSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class ReturnStatementLowerer : public Lowerer {
public:
  explicit ReturnStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::ReturnStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_RETURNSTATEMENTLOWERER_H
