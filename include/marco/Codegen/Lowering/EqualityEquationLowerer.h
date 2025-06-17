#ifndef MARCO_CODEGEN_LOWERING_EQUALITYEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUALITYEQUATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class EqualityEquationLowerer : public Lowerer {
public:
  explicit EqualityEquationLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::EqualityEquation &equation) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_EQUALITYEQUATIONLOWERER_H
