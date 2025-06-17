#ifndef MARCO_CODEGEN_LOWERING_COMPONENTREFERENCELOWERER_H
#define MARCO_CODEGEN_LOWERING_COMPONENTREFERENCELOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class ComponentReferenceLowerer : public Lowerer {
public:
  explicit ComponentReferenceLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::ComponentReference &componentReference) override;

private:
  std::optional<Reference>
  lowerSubscripts(Reference current, const ast::ComponentReferenceEntry &entry,
                  bool isFirst, bool isLast);

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_COMPONENTREFERENCELOWERER_H
