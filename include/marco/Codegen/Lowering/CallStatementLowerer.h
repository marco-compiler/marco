#ifndef MARCO_CODEGEN_LOWERING_CALLSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_CALLSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class CallStatementLowerer : public Lowerer {
public:
  explicit CallStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::CallStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_CALLSTATEMENTLOWERER_H
