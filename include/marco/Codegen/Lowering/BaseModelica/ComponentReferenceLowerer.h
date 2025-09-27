#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_COMPONENTREFERENCELOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_COMPONENTREFERENCELOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class ComponentReferenceLowerer : public Lowerer {
public:
  explicit ComponentReferenceLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::ComponentReference &componentReference) override;

private:
  std::optional<Reference>
  lowerSubscripts(Reference current,
                  const ast::bmodelica::ComponentReferenceEntry &entry,
                  bool isFirst, bool isLast);

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_COMPONENTREFERENCELOWERER_H
