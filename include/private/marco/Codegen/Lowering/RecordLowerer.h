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
      RecordLowerer(LoweringContext* context, BridgeInterface* bridge);

      std::vector<mlir::Operation*> lower(const ast::Record& model);

    protected:
      using Lowerer::lower;

    private:
      void lower(const ast::Member& member);
  };
}

#endif // MARCO_CODEGEN_LOWERING_RECORDLOWERER_H
