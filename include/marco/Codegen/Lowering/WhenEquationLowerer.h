#ifndef MARCO_CODEGEN_LOWERING_WHENEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_WHENEQUATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class WhenEquationLowerer : public Lowerer {
public:
  explicit WhenEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::WhenEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_WHENEQUATIONLOWERER_H
