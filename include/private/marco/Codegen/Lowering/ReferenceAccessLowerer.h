#ifndef MARCO_CODEGEN_LOWERING_REFERENCEACCESSLOWERER_H
#define MARCO_CODEGEN_LOWERING_REFERENCEACCESSLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ReferenceAccessLowerer : public Lowerer
  {
    public:
      ReferenceAccessLowerer(BridgeInterface* bridge);

      Results lower(const ast::ReferenceAccess& referenceAccess) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_REFERENCEACCESSLOWERER_H
