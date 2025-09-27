#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_WHENSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_WHENSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class WhenStatementLowerer : public Lowerer {
public:
  explicit WhenStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::WhenStatement &statement) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_WHENSTATEMENTLOWERER_H
