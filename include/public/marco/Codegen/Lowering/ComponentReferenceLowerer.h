#ifndef MARCO_CODEGEN_LOWERING_COMPONENTREFERENCELOWERER_H
#define MARCO_CODEGEN_LOWERING_COMPONENTREFERENCELOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ComponentReferenceLowerer : public Lowerer
  {
    public:
      explicit ComponentReferenceLowerer(BridgeInterface* bridge);

      std::optional<Results> lower(
          const ast::ComponentReference& componentReference) override;

    private:
      std::optional<Reference> lowerSubscripts(
        Reference current, const ast::ComponentReferenceEntry& entry);

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_COMPONENTREFERENCELOWERER_H
