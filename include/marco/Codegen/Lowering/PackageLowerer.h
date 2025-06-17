#ifndef MARCO_CODEGEN_LOWERING_PACKAGELOWERER_H
#define MARCO_CODEGEN_LOWERING_PACKAGELOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class PackageLowerer : public Lowerer {
public:
  explicit PackageLowerer(BridgeInterface *bridge);

  void declare(const ast::Package &package) override;

  [[nodiscard]] bool declareVariables(const ast::Package &package) override;

  [[nodiscard]] bool lower(const ast::Package &package) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_PACKAGELOWERER_H
