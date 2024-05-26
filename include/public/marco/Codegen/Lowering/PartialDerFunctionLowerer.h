#ifndef MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class PartialDerFunctionLowerer : public Lowerer
  {
    public:
      explicit PartialDerFunctionLowerer(BridgeInterface* bridge);

      void declare(const ast::PartialDerFunction& function) override;

      void declareVariables(const ast::PartialDerFunction& function) override;

      void lower(const ast::PartialDerFunction& function) override;

    protected:
      using Lowerer::declare;
      using Lowerer::declareVariables;
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_PARTIALDERFUNCTIONLOWERER_H
