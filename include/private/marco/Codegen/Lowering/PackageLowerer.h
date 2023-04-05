#ifndef MARCO_CODEGEN_LOWERING_PACKAGELOWERER_H
#define MARCO_CODEGEN_LOWERING_PACKAGELOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class PackageLowerer : public Lowerer
  {
    public:
      PackageLowerer(BridgeInterface* bridge);

      void declare(const ast::Package& package) override;

      void lower(const ast::Package& package) override;

    protected:
      using Lowerer::declare;
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_PACKAGELOWERER_H
