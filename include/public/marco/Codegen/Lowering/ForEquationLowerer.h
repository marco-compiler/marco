#ifndef MARCO_CODEGEN_LOWERING_FOREQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_FOREQUATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ForEquationLowerer : public Lowerer
  {
    public:
      explicit ForEquationLowerer(BridgeInterface* bridge);

      void lower(const ast::ForEquation& equation) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_FOREQUATIONLOWERER_H
