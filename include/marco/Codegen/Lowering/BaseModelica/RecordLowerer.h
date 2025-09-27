#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_RECORDLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_RECORDLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class RecordLowerer : public Lowerer {
public:
  explicit RecordLowerer(BridgeInterface *bridge);

  void declare(const ast::bmodelica::Record &record) override;

  [[nodiscard]] bool
  declareVariables(const ast::bmodelica::Record &record) override;

  [[nodiscard]] bool lower(const ast::bmodelica::Record &record) override;

protected:
  using Lowerer::declare;
  using Lowerer::declareVariables;
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_RECORDLOWERER_H
