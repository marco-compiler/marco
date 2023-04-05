#ifndef MARCO_CODEGEN_LOWERING_TUPLELOWERER_H
#define MARCO_CODEGEN_LOWERING_TUPLELOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class TupleLowerer : public Lowerer
  {
    public:
      TupleLowerer(BridgeInterface* bridge);

      Results lower(const ast::Tuple& tuple) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_TUPLELOWERER_H
