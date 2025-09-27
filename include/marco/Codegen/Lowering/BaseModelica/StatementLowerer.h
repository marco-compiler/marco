#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_STATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_STATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class StatementLowerer : public Lowerer {
public:
  explicit StatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::bmodelica::Statement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_STATEMENTLOWERER_H
