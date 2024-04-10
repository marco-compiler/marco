#ifndef MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
#define MARCO_CODEGEN_LOWERING_RECORDLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class RecordLowerer : public Lowerer
  {
    public:
      explicit RecordLowerer(BridgeInterface* bridge);

      void declare(const ast::Record& record) override;

      __attribute__((warn_unused_result)) bool declareVariables(const ast::Record& record) override;

      __attribute__((warn_unused_result)) bool lower(const ast::Record& record) override;

    protected:
      using Lowerer::declare;
      using Lowerer::declareVariables;
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
