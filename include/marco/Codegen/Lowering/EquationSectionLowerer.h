#ifndef MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class EquationSectionLowerer : public Lowerer {
public:
  explicit EquationSectionLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::EquationSection &equationSection) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_EQUATIONSECTIONLOWERER_H
