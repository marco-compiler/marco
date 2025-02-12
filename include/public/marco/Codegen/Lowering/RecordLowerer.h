#ifndef MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
#define MARCO_CODEGEN_LOWERING_RECORDLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class RecordLowerer : public Lowerer {
public:
  explicit RecordLowerer(BridgeInterface *bridge);

  void declare(const ast::Record &record) override;

  [[nodiscard]] bool declareVariables(const ast::Record &record) override;

  [[nodiscard]] bool lower(const ast::Record &record) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
