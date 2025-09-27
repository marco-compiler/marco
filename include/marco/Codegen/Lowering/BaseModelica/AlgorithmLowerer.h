#ifndef MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H
#define MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class AlgorithmLowerer : public Lowerer {
public:
  explicit AlgorithmLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::bmodelica::Algorithm &algorithm) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H
