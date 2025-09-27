#ifndef MARCO_CODEGEN_LOWERING_TUPLELOWERER_H
#define MARCO_CODEGEN_LOWERING_TUPLELOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class TupleLowerer : public Lowerer {
public:
  explicit TupleLowerer(BridgeInterface *bridge);

  std::optional<Results> lower(const ast::bmodelica::Tuple &tuple) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_TUPLELOWERER_H
