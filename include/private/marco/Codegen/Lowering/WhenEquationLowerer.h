#ifndef MARCO_CODEGEN_LOWERING_WHENEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_WHENEQUATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class WhenEquationLowerer : public Lowerer
  {
    public:
      WhenEquationLowerer(BridgeInterface* bridge);

      void lower(const ast::WhenEquation& equation) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_WHENEQUATIONLOWERER_H
