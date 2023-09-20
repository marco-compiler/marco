#ifndef MARCO_CODEGEN_LOWERING_SUBSCRIPTLOWERER_H
#define MARCO_CODEGEN_LOWERING_SUBSCRIPTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class SubscriptLowerer : public Lowerer
  {
    public:
      SubscriptLowerer(BridgeInterface* bridge);

      Results lower(const ast::Subscript& subscript) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_SUBSCRIPTLOWERER_H
