#ifndef MARCO_CODEGEN_LOWERING_IFEQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_IFEQUATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class IfEquationLowerer : public Lowerer
  {
    public:
      explicit IfEquationLowerer(BridgeInterface* bridge);

      [[nodiscard]] virtual bool lower(const ast::IfEquation& equation) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_IFEQUATIONLOWERER_H
