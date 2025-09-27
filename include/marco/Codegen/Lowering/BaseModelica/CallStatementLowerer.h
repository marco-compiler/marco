#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class CallStatementLowerer : public Lowerer {
public:
  explicit CallStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::CallStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_CALLSTATEMENTLOWERER_H
