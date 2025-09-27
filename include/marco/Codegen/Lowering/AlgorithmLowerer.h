#ifndef MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H
#define MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class AlgorithmLowerer : public Lowerer {
public:
  explicit AlgorithmLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::bmodelica::Algorithm &algorithm) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H
