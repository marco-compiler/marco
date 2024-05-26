#ifndef MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H
#define MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class AlgorithmLowerer : public Lowerer
  {
    public:
      explicit AlgorithmLowerer(BridgeInterface* bridge);

      void lower(const ast::Algorithm& algorithm) override;

    protected:
      using Lowerer::lower;

  };
}

#endif // MARCO_CODEGEN_LOWERING_ALGORITHMLOWERER_H
