#ifndef MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class EquationLowerer : public Lowerer
  {
    public:
      explicit EquationLowerer(BridgeInterface* bridge);

      void lower(const ast::Equation& equation) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H
