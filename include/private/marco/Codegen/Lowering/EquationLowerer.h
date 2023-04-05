#ifndef MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class EquationLowerer : public Lowerer
  {
    public:
      EquationLowerer(BridgeInterface* bridge);

      void lower(
          const ast::Equation& equation,
          bool initialEquation) override;

      void lower(
          const ast::ForEquation& forEquation,
          bool initialEquation) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_EQUATIONLOWERER_H
