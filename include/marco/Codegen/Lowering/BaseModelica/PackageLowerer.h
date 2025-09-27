#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_PACKAGELOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_PACKAGELOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class PackageLowerer : public Lowerer {
public:
  explicit PackageLowerer(BridgeInterface *bridge);

  void declare(const ast::bmodelica::Package &package) override;

  [[nodiscard]] bool
  declareVariables(const ast::bmodelica::Package &package) override;

  [[nodiscard]] bool lower(const ast::bmodelica::Package &package) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_PACKAGELOWERER_H
