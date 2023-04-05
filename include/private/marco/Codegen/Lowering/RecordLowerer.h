#ifndef MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
#define MARCO_CODEGEN_LOWERING_RECORDLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class RecordLowerer : public Lowerer
  {
    public:
      RecordLowerer(BridgeInterface* bridge);

      void declare(const ast::Record& record) override;

      void lower(const ast::Record& record) override;

    protected:
      using Lowerer::declare;
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
