#ifndef MARCO_CODEGEN_LOWERING_ARRAYLOWERER_H
#define MARCO_CODEGEN_LOWERING_ARRAYLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ArrayLowerer : public Lowerer
  {
    public:
      ArrayLowerer(BridgeInterface* bridge);

      Results lower(const ast::Array& array) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_ARRAYLOWERER_H
